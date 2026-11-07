// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <bit>
#include <cstdio>
#include <optional>
#include <set>
#include <utility>

#include "shader_recompiler/frontend/copy_shader.h"
#include "shader_recompiler/frontend/decode.h"
#include "shader_recompiler/ir/attribute.h"

namespace Shader {

namespace {

/// gow3 (stream-out): a dword whose origin the parser knows, in StreamOutSharp terms.
struct ScalarLocation {
    bool valid{};
    StreamOutSharp at{}; ///< for hops > 0, offsets[hops - 1] is this dword

    ScalarLocation Next() const {
        ScalarLocation next = *this;
        if (at.hops == 0) {
            ++next.at.reg;
        } else {
            ++next.at.offsets[at.hops - 1];
        }
        return next;
    }

    bool operator==(const ScalarLocation&) const = default;
};

constexpr u32 NumUserSgprs = 16;
constexpr u32 UnresolvedHops = 3; ///< StreamOutSharp::hops of a V# the parser could not trace

using SgprConstants = std::array<std::optional<u32>, 128>;

std::optional<u32> ConstantOf(const Gcn::InstOperand& op, const SgprConstants& sgpr_constants) {
    switch (op.field) {
    case Gcn::OperandField::ConstZero:
        return 0u;
    case Gcn::OperandField::SignedConstIntPos:
        return op.code - Gcn::OperandFieldRange::SignedConstIntPosMin + 1;
    case Gcn::OperandField::SignedConstIntNeg:
        return static_cast<u32>(-static_cast<s32>(op.code - Gcn::OperandFieldRange::SignedConstIntNegMin + 1));
    case Gcn::OperandField::LiteralConst:
        return op.code;
    case Gcn::OperandField::ScalarGPR:
        return op.code < sgpr_constants.size() ? sgpr_constants[op.code] : std::nullopt;
    default:
        return std::nullopt;
    }
}

u32 Width(const Gcn::InstOperand& op) {
    switch (op.type) {
    case Gcn::ScalarType::Uint64:
    case Gcn::ScalarType::Sint64:
    case Gcn::ScalarType::Float64:
        return 2;
    default:
        return 1;
    }
}

/// S_LOAD_DWORD*: loads through a 64-bit pointer.
bool IsScalarLoad(Gcn::Opcode opcode) {
    return opcode >= Gcn::Opcode::S_LOAD_DWORD && opcode <= Gcn::Opcode::S_LOAD_DWORDX16;
}

/// Any SMRD load (S_LOAD_DWORD*, S_BUFFER_LOAD_DWORD*): writes `smrd.count` registers.
bool IsSmrdLoad(Gcn::Opcode opcode) {
    return opcode >= Gcn::Opcode::S_LOAD_DWORD && opcode <= Gcn::Opcode::S_BUFFER_LOAD_DWORDX16;
}

/// Number of dwords a buffer store writes, 0 for other instructions.
u32 StoreDwords(const Gcn::GcnInst& inst) {
    using Gcn::Opcode;
    if ((inst.opcode >= Opcode::BUFFER_STORE_DWORD && inst.opcode <= Opcode::BUFFER_STORE_DWORDX3) ||
        (inst.opcode >= Opcode::BUFFER_STORE_FORMAT_X &&
         inst.opcode <= Opcode::BUFFER_STORE_FORMAT_XYZW)) {
        return static_cast<u32>(inst.control.mubuf.count);
    }
    if (inst.opcode >= Opcode::TBUFFER_STORE_FORMAT_X &&
        inst.opcode <= Opcode::TBUFFER_STORE_FORMAT_XYZW) {
        return static_cast<u32>(inst.control.mtbuf.count);
    }
    return 0;
}

} // namespace

CopyShaderData ParseCopyShader(std::span<const u32> code) {
    Gcn::GcnCodeSlice code_slice{code.data(), code.data() + code.size()};
    Gcn::GcnDecodeContext decoder;

    constexpr u32 token_mov_vcchi = 0xBEEB03FF;
    ASSERT_MSG(code[0] == token_mov_vcchi, "First instruction is not s_mov_b32 vcc_hi, #imm");

    std::array<s32, 64> offsets{};
    offsets.fill(-1);

    std::array<s32, 256> sources{};
    sources.fill(-1);

    // gow3 (stream-out): where scalar registers come from (to find the V# of each store), their
    // constant values, the ring dword each vector register holds, and constants added to store
    // addresses (attribute offsets folded into the address register).
    std::array<ScalarLocation, 128> sgpr_locations{};
    for (u32 i = 0; i < NumUserSgprs; ++i) {
        sgpr_locations[i] = {true, {i, 0, {}}};
    }
    SgprConstants sgpr_constants{};
    std::array<s32, 256> vgpr_ring{};
    vgpr_ring.fill(-1);
    std::array<u32, 256> vgpr_delta{};
    std::set<u32> ring_loads;

    CopyShaderData data{};
    auto last_attr{IR::Attribute::Position0};
    while (!code_slice.atEnd()) {
        auto inst = decoder.decodeInstruction(code_slice);
        switch (inst.opcode) {
        case Gcn::Opcode::S_LSHL_B32: {
            ASSERT(inst.src[0].field == Gcn::OperandField::SignedConstIntPos &&
                   inst.src[1].field == Gcn::OperandField::SignedConstIntPos);
            sources[inst.dst[0].code] =
                (inst.src[0].code - Gcn::OperandFieldRange::SignedConstIntPosMin + 1)
                << (inst.src[1].code - Gcn::OperandFieldRange::SignedConstIntPosMin + 1);
            break;
        }
        case Gcn::Opcode::S_MOVK_I32: {
            sources[inst.dst[0].code] = inst.control.sopk.simm;
            break;
        }
        case Gcn::Opcode::S_MOV_B32: {
            sources[inst.dst[0].code] = inst.src[0].code;
            break;
        }
        case Gcn::Opcode::S_ADDK_I32: {
            sources[inst.dst[0].code] += inst.control.sopk.simm;
            break;
        }
        case Gcn::Opcode::S_BFM_B32: {
            ASSERT(inst.src[0].field == Gcn::OperandField::SignedConstIntPos &&
                   inst.src[1].field == Gcn::OperandField::SignedConstIntPos);
            const auto src0 = inst.src[0].code - Gcn::OperandFieldRange::SignedConstIntPosMin + 1;
            const auto src1 = inst.src[1].code - Gcn::OperandFieldRange::SignedConstIntPosMin + 1;
            sources[inst.dst[0].code] = ((1 << src0) - 1) << src1;
            break;
        }
        case Gcn::Opcode::EXP: {
            const auto& exp = inst.control.exp;
            const IR::Attribute semantic = static_cast<IR::Attribute>(exp.target);
            for (int i = 0; i < inst.src_count; ++i) {
                if ((exp.en & (1 << i)) == 0) {
                    continue;
                }
                const auto ofs = offsets[inst.src[i].code];
                if (ofs != -1) {
                    data.attr_map[ofs] = {semantic, i};
                    if (semantic > last_attr) {
                        last_attr = semantic;
                    }
                }
            }
            break;
        }
        case Gcn::Opcode::BUFFER_LOAD_DWORD: {
            ASSERT_MSG(inst.src[1].code < offsets.size(),
                       "offsets array for geometry shaders is too short");
            offsets[inst.src[1].code] = inst.control.mubuf.offset;
            if (inst.src[3].field != Gcn::OperandField::ConstZero) {
                const u32 index = inst.src[3].code;
                ASSERT(sources[index] != -1);
                offsets[inst.src[1].code] += sources[index];
            }
            data.num_comps++;
            break;
        }
        default:
            break;
        }

        // gow3 (stream-out): register tracking, after the export mapping above.
        const auto opcode = inst.opcode;
        for (u32 i = 0; i < inst.dst_count; ++i) {
            const auto& dst = inst.dst[i];
            if (dst.field == Gcn::OperandField::ScalarGPR) {
                const u32 width = IsSmrdLoad(opcode) ? inst.control.smrd.count : Width(dst);
                for (u32 w = 0; w < width && dst.code + w < sgpr_locations.size(); ++w) {
                    sgpr_locations[dst.code + w] = {};
                    sgpr_constants[dst.code + w].reset();
                }
            } else if (dst.field == Gcn::OperandField::VectorGPR && dst.code < vgpr_ring.size()) {
                vgpr_ring[dst.code] = -1;
                vgpr_delta[dst.code] = 0;
            }
        }
        switch (opcode) {
        case Gcn::Opcode::S_MOV_B32:
            if (inst.src[0].field == Gcn::OperandField::ScalarGPR &&
                inst.src[0].code < sgpr_locations.size()) {
                sgpr_locations[inst.dst[0].code] = sgpr_locations[inst.src[0].code];
            }
            sgpr_constants[inst.dst[0].code] = ConstantOf(inst.src[0], sgpr_constants);
            break;
        case Gcn::Opcode::S_MOV_B64:
            if (inst.src[0].field == Gcn::OperandField::ScalarGPR &&
                inst.src[0].code + 1 < sgpr_locations.size() &&
                inst.dst[0].code + 1 < sgpr_locations.size()) {
                sgpr_locations[inst.dst[0].code] = sgpr_locations[inst.src[0].code];
                sgpr_locations[inst.dst[0].code + 1] = sgpr_locations[inst.src[0].code + 1];
            }
            break;
        case Gcn::Opcode::S_MOVK_I32:
            sgpr_constants[inst.dst[0].code] =
                static_cast<u32>(static_cast<s32>(static_cast<s16>(inst.control.sopk.simm)));
            break;
        case Gcn::Opcode::V_MOV_B32:
            if (inst.src[0].field == Gcn::OperandField::VectorGPR &&
                inst.dst[0].code < vgpr_ring.size() && inst.src[0].code < vgpr_ring.size()) {
                vgpr_ring[inst.dst[0].code] = vgpr_ring[inst.src[0].code];
                vgpr_delta[inst.dst[0].code] = vgpr_delta[inst.src[0].code];
            }
            break;
        case Gcn::Opcode::V_ADD_I32: {
            // An attribute offset folded into a store's address register.
            for (u32 i = 0; i < 2; ++i) {
                const auto& reg = inst.src[i];
                const auto constant = ConstantOf(inst.src[i ^ 1], sgpr_constants);
                if (reg.field == Gcn::OperandField::VectorGPR && constant &&
                    reg.code < vgpr_delta.size() && inst.dst[0].code < vgpr_delta.size()) {
                    vgpr_delta[inst.dst[0].code] = vgpr_delta[reg.code] + *constant;
                    break;
                }
            }
            break;
        }
        case Gcn::Opcode::BUFFER_LOAD_DWORD:
            vgpr_ring[inst.src[1].code] = offsets[inst.src[1].code];
            ring_loads.insert(static_cast<u32>(offsets[inst.src[1].code]));
            break;
        default:
            if (IsScalarLoad(opcode)) {
                const u32 sbase = inst.src[0].code * 2;
                const auto& pointer = sgpr_locations[sbase];
                const auto& pointer_hi = sgpr_locations[sbase + 1];
                std::optional<u32> dword_offset;
                if (inst.control.smrd.imm) {
                    dword_offset = inst.control.smrd.offset;
                } else if (inst.control.smrd.offset == 0xFF) {
                    dword_offset = inst.src[1].code;
                } else if (inst.control.smrd.offset < sgpr_constants.size() &&
                           sgpr_constants[inst.control.smrd.offset]) {
                    dword_offset = *sgpr_constants[inst.control.smrd.offset] / 4;
                }
                if (pointer.valid && pointer_hi == pointer.Next() && pointer.at.hops < 2 &&
                    dword_offset) {
                    for (u32 i = 0; i < inst.control.smrd.count &&
                                    inst.dst[0].code + i < sgpr_locations.size();
                         ++i) {
                        ScalarLocation loaded{true, pointer.at};
                        loaded.at.offsets[pointer.at.hops] = *dword_offset + i;
                        ++loaded.at.hops;
                        sgpr_locations[inst.dst[0].code + i] = loaded;
                    }
                }
            } else if (const u32 dwords = StoreDwords(inst)) {
                const bool is_mtbuf = inst.opcode >= Gcn::Opcode::TBUFFER_STORE_FORMAT_X &&
                                      inst.opcode <= Gcn::Opcode::TBUFFER_STORE_FORMAT_XYZW;
                const bool offen = is_mtbuf ? inst.control.mtbuf.offen : inst.control.mubuf.offen;
                const bool idxen = is_mtbuf ? inst.control.mtbuf.idxen : inst.control.mubuf.idxen;
                u32 offset = is_mtbuf ? inst.control.mtbuf.offset : inst.control.mubuf.offset;
                if (offen) {
                    // With an index as well, the offset is the second address register.
                    const u32 offset_reg = inst.src[0].code + (idxen ? 1 : 0);
                    offset += offset_reg < vgpr_delta.size() ? vgpr_delta[offset_reg] : 0;
                }
                offset += ConstantOf(inst.src[3], sgpr_constants).value_or(0);
                // The V#: four consecutive dwords of one origin.
                const u32 srsrc = inst.src[2].code * 4;
                ScalarLocation location = sgpr_locations[srsrc];
                bool traced = location.valid;
                for (u32 i = 1; i < 4 && traced; ++i) {
                    location = location.Next();
                    traced = sgpr_locations[srsrc + i] == location;
                }
                StreamOutSharp sharp = sgpr_locations[srsrc].at;
                if (!traced) {
                    sharp = {srsrc, UnresolvedHops, {}};
                }
                auto it = std::ranges::find(data.so_sharps, sharp);
                if (it == data.so_sharps.end()) {
                    data.so_sharps.push_back(sharp);
                    it = data.so_sharps.end() - 1;
                }
                const u32 sharp_index = static_cast<u32>(it - data.so_sharps.begin());
                for (u32 i = 0; i < dwords; ++i) {
                    const u32 vdata = inst.src[1].code + i;
                    if (vdata < vgpr_ring.size() && vgpr_ring[vdata] >= 0) {
                        data.so_stores.push_back(
                            {static_cast<u32>(vgpr_ring[vdata]), sharp_index, offset + i * 4});
                    }
                }
            }
            break;
        }
    }

    if (!IsPosition(last_attr)) {
        data.num_attrs = static_cast<u32>(last_attr) - static_cast<u32>(IR::Attribute::Param0) + 1;
    }
    if (data.attr_map.size() > 1) {
        const auto it = data.attr_map.begin();
        const u32 comp_stride = std::next(it)->first - it->first;
        data.output_vertices = comp_stride / 64;
    } else if (ring_loads.size() > 1) {
        // gow3 (stream-out): a copy shader exporting at most one dword (all its other ring
        // reads are streamed out): the component stride from all ring reads instead.
        const u32 comp_stride = *std::next(ring_loads.begin()) - *ring_loads.begin();
        data.output_vertices = comp_stride / 64;
    }

    // gow3 (stream-out): V#s in table order (MapStreamOutBuffers relies on it).
    if (data.so_sharps.size() > 1) {
        auto sorted = data.so_sharps;
        std::ranges::sort(sorted);
        for (auto& store : data.so_stores) {
            store.sharp = static_cast<u32>(std::ranges::find(sorted, data.so_sharps[store.sharp]) -
                                           sorted.begin());
        }
        data.so_sharps = std::move(sorted);
    }

    return data;
}

std::array<s32, 4> MapStreamOutBuffers(const CopyShaderData& data, u32 buffer_mask) {
    std::array<s32, 4> buffers;
    buffers.fill(-1);
    u32 remaining = buffer_mask & 0xF;
    for (u32 i = 0; i < data.so_sharps.size() && i < buffers.size() && remaining; ++i) {
        const u32 buffer = static_cast<u32>(std::countr_zero(remaining));
        buffers[i] = static_cast<s32>(buffer);
        remaining &= remaining - 1;
    }
    return buffers;
}

std::vector<XfbOutput> BuildXfbOutputs(const CopyShaderData& data, u32 buffer_mask,
                                       const std::array<u32, 4>& strides, u32 first_param) {
    const auto buffers = MapStreamOutBuffers(data, buffer_mask);
    std::array<std::map<u32, u32>, 4> by_offset; // byte offset -> ring dword, per buffer
    u32 dropped = 0;
    for (const auto& store : data.so_stores) {
        const s32 buffer = store.sharp < buffers.size() ? buffers[store.sharp] : -1;
        if (buffer < 0 || store.byte_offset % 4 != 0 ||
            store.byte_offset + 4 > strides[buffer] * 4) {
            ++dropped;
            continue;
        }
        by_offset[buffer].emplace(store.byte_offset, store.ring_offset);
    }
    std::vector<XfbOutput> outputs;
    for (u32 buffer = 0; buffer < by_offset.size(); ++buffer) {
        XfbOutput* run = nullptr;
        for (const auto& [byte_offset, ring_offset] : by_offset[buffer]) {
            if (run && run->num_components < 4 &&
                run->byte_offset + run->num_components * 4 == byte_offset) {
                run->ring_offsets[run->num_components++] = ring_offset;
                continue;
            }
            const u32 param = first_param + static_cast<u32>(outputs.size());
            if (param >= IR::NumParams) {
                ++dropped;
                run = nullptr;
                continue;
            }
            run = &outputs.emplace_back(XfbOutput{param, buffer, byte_offset, 1, {ring_offset}});
        }
    }
    if (dropped) {
        static bool reported = false;
        if (!std::exchange(reported, true)) {
            std::printf("Stream-out: %u copy shader stores left out of the capture (no buffer, "
                        "past the vertex stride or out of output slots)\n",
                        dropped);
            std::fflush(stdout);
        }
    }
    return outputs;
}

} // namespace Shader
