// SPDX-License-Identifier: GPL-2.0-or-later
// gow3 (stream-out): ParseCopyShader's stream-out tracking and BuildXfbOutputs on hand-assembled
// GCN copy shaders (vk_stream_out.h). Target copy-shader-streamout-test; or standalone with stub
// fmt/logging headers and -DCOPY_SHADER_TEST_STANDALONE, compiling copy_shader.cpp, decode.cpp,
// format.cpp and instruction.cpp alongside.
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "shader_recompiler/frontend/copy_shader.h"

#ifdef COPY_SHADER_TEST_STANDALONE
void assert_fail_impl() {
    std::fprintf(stderr, "assert failed\n");
    std::abort();
}
void unreachable_impl() {
    std::fprintf(stderr, "unreachable\n");
    std::abort();
}
#endif

using u32 = unsigned;
static std::vector<u32> code;
static int failures = 0;
#define CHECK(c)                                                                                   \
    do {                                                                                           \
        if (!(c)) {                                                                                \
            std::printf("FAIL line %d: %s\n", __LINE__, #c);                                       \
            ++failures;                                                                            \
        }                                                                                          \
    } while (0)

// GCN (GFX7) encodings of the few instructions a copy shader needs.
static void mov_vcchi() { // s_mov_b32 vcc_hi, literal: the PS4 shader header
    code.push_back(0xBEEB03FF);
    code.push_back(0x10);
}
static void s_movk(u32 sdst, u32 imm) {
    code.push_back((0b1011u << 28) | (sdst << 16) | (imm & 0xFFFF));
}
static void s_load(u32 op, u32 sdst, u32 sbase, u32 off) {
    code.push_back((0b11000u << 27) | (op << 22) | (sdst << 15) | ((sbase / 2) << 9) | (1u << 8) |
                   off);
}
static void mubuf(u32 op, u32 vdata, u32 vaddr, u32 srsrc, u32 soffset, u32 offset, bool offen,
                  bool glc, bool slc) {
    code.push_back((0b111000u << 26) | (op << 18) | (glc ? 1u << 14 : 0) | (offen ? 1u << 12 : 0) |
                   (offset & 0xFFF));
    code.push_back(vaddr | (vdata << 8) | ((srsrc / 4) << 16) | (slc ? 1u << 22 : 0) |
                   (soffset << 24));
}
static void v_add_i32_const(u32 vdst, u32 inline_const, u32 vsrc1) {
    code.push_back((37u << 25) | (vdst << 17) | (vsrc1 << 9) | (128 + inline_const));
}
static void v_mov(u32 vdst, u32 vsrc) {
    code.push_back((0b0111111u << 25) | (vdst << 17) | (1u << 9) | (256 + vsrc));
}
static void exp(u32 tgt, u32 en, u32 v0, u32 v1, u32 v2, u32 v3, bool done) {
    code.push_back((0b111110u << 26) | (done ? 1u << 11 : 0) | (tgt << 4) | en);
    code.push_back(v0 | (v1 << 8) | (v2 << 16) | (v3 << 24));
}
static void s_endpgm() {
    code.push_back(0xBF810000);
}

constexpr u32 LOAD = 12, STORE = 28, STORE_X4 = 30, ZERO = 128;

/// Position (ring 0, 0x400, 0x800, 0xC00 into v1-v4) and one parameter dword (0x1000 into v6).
static void ring_loads() {
    mubuf(LOAD, 1, 0, 4, ZERO, 0, true, true, true);
    s_movk(12, 0x400);
    mubuf(LOAD, 2, 0, 4, 12, 0, true, true, true);
    s_movk(13, 0x800);
    mubuf(LOAD, 3, 0, 4, 13, 0, true, true, true);
    s_movk(14, 0xC00);
    mubuf(LOAD, 4, 0, 4, 14, 0, true, true, true);
    s_movk(15, 0x1000);
    mubuf(LOAD, 6, 0, 4, 15, 0, true, true, true);
}

int main() {
    // All five dwords streamed out through a V# loaded from the table at user data s[0:1]
    // (dword 4); param0.x 16 bytes in (an address register + 16), through a copied register.
    mov_vcchi();
    ring_loads();
    v_mov(7, 6);
    s_load(2, 8, 0, 4); // s_load_dwordx4 s[8:11], s[0:1], 4
    mubuf(STORE_X4, 1, 8, 8, ZERO, 0, true, false, false);
    v_add_i32_const(9, 16, 8);
    mubuf(STORE, 7, 9, 8, ZERO, 0, true, false, false);
    exp(12, 0xF, 1, 2, 3, 4, true);
    exp(32, 0x1, 6, 0, 0, 0, false);
    s_endpgm();

    const auto data = Shader::ParseCopyShader(code);
    CHECK(data.attr_map.size() == 5);
    CHECK(data.num_attrs == 1);
    CHECK(data.output_vertices == 16);
    CHECK(data.so_stores.size() == 5);
    CHECK(data.so_sharps.size() == 1);
    CHECK(!data.so_sharps.empty() && data.so_sharps[0].reg == 0 && data.so_sharps[0].hops == 1 &&
          data.so_sharps[0].offsets[0] == 4);
    const u32 rings[5] = {0, 0x400, 0x800, 0xC00, 0x1000};
    for (u32 i = 0; i < 5 && i < data.so_stores.size(); ++i) {
        CHECK(data.so_stores[i].ring_offset == rings[i]);
        CHECK(data.so_stores[i].byte_offset == i * 4);
    }
    const auto outputs = Shader::BuildXfbOutputs(data, 0x1, {5, 0, 0, 0}, data.num_attrs);
    CHECK(outputs.size() == 2);
    if (outputs.size() == 2) {
        CHECK(outputs[0].param == 1 && outputs[0].buffer == 0 && outputs[0].byte_offset == 0 &&
              outputs[0].num_components == 4 && outputs[0].ring_offsets[3] == 0xC00);
        CHECK(outputs[1].param == 2 && outputs[1].byte_offset == 16 &&
              outputs[1].num_components == 1 && outputs[1].ring_offsets[0] == 0x1000);
    }
    // Buffer 2 enabled instead: XfbBuffer 2.
    const auto on2 = Shader::BuildXfbOutputs(data, 0x4, {0, 0, 5, 0}, data.num_attrs);
    CHECK(on2.size() == 2 && on2[0].buffer == 2);
    // A vertex stride of 4 dwords leaves param0.x out.
    const auto narrow = Shader::BuildXfbOutputs(data, 0x1, {4, 0, 0, 0}, data.num_attrs);
    CHECK(narrow.size() == 1 && narrow[0].num_components == 4);
    CHECK(Shader::BuildXfbOutputs(data, 0, {}, data.num_attrs).empty());

    // V# straight from user data s[0:3]; the streamed parameter is not exported.
    code.clear();
    mov_vcchi();
    ring_loads();
    mubuf(STORE, 6, 8, 0, ZERO, 8, true, false, false);
    exp(12, 0xF, 1, 2, 3, 4, true);
    s_endpgm();
    const auto direct = Shader::ParseCopyShader(code);
    CHECK(direct.so_sharps.size() == 1 && direct.so_sharps[0].hops == 0 &&
          direct.so_sharps[0].reg == 0);
    CHECK(direct.so_stores.size() == 1 && direct.so_stores[0].ring_offset == 0x1000 &&
          direct.so_stores[0].byte_offset == 8);
    CHECK(direct.num_attrs == 0);
    CHECK(direct.output_vertices == 16);
    const auto direct_out = Shader::BuildXfbOutputs(direct, 0x1, {4, 0, 0, 0}, direct.num_attrs);
    CHECK(direct_out.size() == 1 && direct_out[0].param == 0 && direct_out[0].byte_offset == 8);

    // Two V#s, met in the opposite of table order: the lower table entry takes the lower buffer.
    code.clear();
    mov_vcchi();
    ring_loads();
    s_load(2, 8, 0, 4);  // table dword 4
    s_load(2, 16, 0, 0); // table dword 0
    mubuf(STORE, 1, 8, 8, ZERO, 0, true, false, false);
    mubuf(STORE, 6, 8, 16, ZERO, 0, true, false, false);
    exp(12, 0xF, 1, 2, 3, 4, true);
    s_endpgm();
    const auto two = Shader::ParseCopyShader(code);
    CHECK(two.so_sharps.size() == 2 && two.so_sharps[0].offsets[0] == 0 &&
          two.so_sharps[1].offsets[0] == 4);
    const auto two_out = Shader::BuildXfbOutputs(two, 0x3, {1, 1, 0, 0}, two.num_attrs);
    CHECK(two_out.size() == 2);
    for (const auto& output : two_out) {
        // Buffer 0 (table dword 0) gets ring 0x1000, buffer 1 (dword 4) ring 0.
        CHECK(output.ring_offsets[0] == (output.buffer == 0 ? 0x1000u : 0u));
    }

    // A copy shader without stores parses as before.
    code.clear();
    mov_vcchi();
    ring_loads();
    exp(12, 0xF, 1, 2, 3, 4, true);
    exp(32, 0x1, 6, 0, 0, 0, false);
    s_endpgm();
    const auto plain = Shader::ParseCopyShader(code);
    CHECK(plain.so_stores.empty() && plain.so_sharps.empty() && plain.num_attrs == 1 &&
          plain.output_vertices == 16 && plain.attr_map.size() == 5);

    std::printf(failures ? "%d FAILURES\n" : "copy shader stream-out: all checks passed\n",
                failures);
    return failures ? 1 : 0;
}
