// SPDX-License-Identifier: GPL-2.0-or-later
// gow3 (Killzone): GCN stream-out through VK_EXT_transform_feedback (see vk_stream_out.h).

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <span>

#include "core/memory.h"
#include "video_core/renderer_vulkan/vk_stream_out.h"
#include "video_core/amdgpu/pm4_cmds.h"
#include "video_core/amdgpu/resource.h"
#include "video_core/buffer_cache/buffer.h"
#include "video_core/buffer_cache/buffer_cache.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_runtime.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/texture_cache/texture_cache.h"

namespace Vulkan {

namespace {

constexpr u64 CounterBytes = 4 * sizeof(u32);

/// Reads `size` bytes of guest memory, or false when they are not mapped.
bool ReadGuest(Core::MemoryManager* memory, VAddr address, void* out, u64 size) {
    if (address == 0 || memory->ClampRangeSize(address, size) != size) {
        return false;
    }
    std::memcpy(out, reinterpret_cast<const void*>(address), size);
    return true;
}

/// The V# a copy shader store uses, read the way the shader reads it (copy_shader.h).
std::optional<AmdGpu::Buffer> ResolveSharp(const Shader::StreamOutSharp& sharp,
                                           const AmdGpu::ShaderProgram& program,
                                           Core::MemoryManager* memory) {
    const u32 num_user_regs = program.settings.num_user_regs;
    AmdGpu::Buffer buffer{};
    if (sharp.hops == 0) {
        if (sharp.reg + 4 > num_user_regs) {
            return std::nullopt;
        }
        std::memcpy(&buffer, &program.user_data[sharp.reg], sizeof(buffer));
        return buffer;
    }
    if (sharp.hops > 2 || sharp.reg + 2 > num_user_regs) {
        return std::nullopt;
    }
    VAddr pointer =
        u64(program.user_data[sharp.reg]) | u64(program.user_data[sharp.reg + 1]) << 32;
    for (u32 hop = 1; hop < sharp.hops; ++hop) {
        u64 next{};
        if (!ReadGuest(memory, pointer + u64(sharp.offsets[hop - 1]) * 4, &next, sizeof(next))) {
            return std::nullopt;
        }
        pointer = next & ((u64{1} << 48) - 1);
    }
    if (!ReadGuest(memory, pointer + u64(sharp.offsets[sharp.hops - 1]) * 4, &buffer,
                   sizeof(buffer))) {
        return std::nullopt;
    }
    return buffer;
}

void PrintCode(const char* name, const AmdGpu::ShaderProgram& program) {
    const auto* code = program.Address<const u32*>();
    if (!code) {
        std::printf("Stream-out:   %s: no program\n", name);
        return;
    }
    const auto params = AmdGpu::GetParams(program);
    std::printf("Stream-out:   %s %#llx hash %016llx, %zu dwords, %u user data:", name,
                static_cast<unsigned long long>(reinterpret_cast<u64>(code)),
                static_cast<unsigned long long>(params.hash), params.code.size(),
                u32(program.settings.num_user_regs));
    for (u32 i = 0; i < program.settings.num_user_regs; ++i) {
        std::printf(" %08x", program.user_data[i]);
    }
    std::printf("\n");
    for (size_t i = 0; i < params.code.size(); ++i) {
        if (i % 8 == 0) {
            std::printf("Stream-out:   %s %04zx:", name, i * 4);
        }
        std::printf(" %08x", params.code[i]);
        if (i % 8 == 7 || i + 1 == params.code.size()) {
            std::printf("\n");
        }
    }
}

u64 Mix(u64 seed, u64 value) {
    return seed ^ (value + 0x9E3779B97F4A7C15ull + (seed << 6) + (seed >> 2));
}

/// Stream-out buffers (stream 0) and their vertex strides, as the geometry shader is specialized
/// (vk_pipeline_cache.cpp: BuildRuntimeInfo).
std::pair<u32, std::array<u32, 4>> BuffersOf(const AmdGpu::Regs& regs) {
    const u32 mask = regs.vgt_strmout_buffer_config.stream_0_buf_en;
    std::array<u32, 4> strides{};
    for (u32 b = 0; b < 4; ++b) {
        strides[b] = (mask >> b) & 1 ? regs.vgt_strmout_buffers[b].vtx_stride : 0;
    }
    return {mask, strides};
}

} // namespace

std::optional<bool> StreamOutPipelineUsable(const AmdGpu::Regs& regs, bool cached_only) {
    static std::mutex mutex;
    static std::unordered_map<u64, bool> verdicts;
    const AmdGpu::ShaderProgram* programs[] = {&regs.vs_program, &regs.gs_program,
                                               &regs.es_program};
    for (const auto* program : programs) {
        if (!program->Address<const u32*>()) {
            return false; // refused later anyway (bind_stage): no program
        }
    }
    const auto [mask, strides] = BuffersOf(regs);
    u64 key = 0;
    for (const auto* program : programs) {
        key = Mix(key, AmdGpu::GetParams(*program).hash);
    }
    key = Mix(key, mask);
    for (const u32 stride : strides) {
        key = Mix(key, stride);
    }
    std::scoped_lock lock{mutex};
    if (const auto it = verdicts.find(key); it != verdicts.end()) {
        return it->second;
    }
    if (cached_only) {
        return std::nullopt;
    }
    // The code first: it is in the log even when the parse below goes wrong.
    std::printf("Stream-out: new pipeline. VGT_STRMOUT_CONFIG %#x, BUFFER_CONFIG %#x, strides "
                "%u/%u/%u/%u dwords, buffer sizes %u/%u/%u/%u dwords\n",
                regs.vgt_strmout_config.raw,
                *reinterpret_cast<const u32*>(&regs.vgt_strmout_buffer_config), strides[0],
                strides[1], strides[2], strides[3], regs.vgt_strmout_buffers[0].size,
                regs.vgt_strmout_buffers[1].size, regs.vgt_strmout_buffers[2].size,
                regs.vgt_strmout_buffers[3].size);
    PrintCode("copy", regs.vs_program);
    PrintCode("gs", regs.gs_program);
    PrintCode("es", regs.es_program);
    std::fflush(stdout);

    const auto params = AmdGpu::GetParams(regs.vs_program);
    const auto data = Shader::ParseCopyShader(params.code);
    u32 dropped = 0;
    std::vector<Shader::XfbOutput> outputs;
    if (data.failure.empty() && data.so_failure.empty()) {
        outputs = Shader::BuildXfbOutputs(data, mask, strides, data.num_attrs, &dropped);
    }
    char reason[320]{};
    if (!data.failure.empty()) {
        std::snprintf(reason, sizeof(reason), "the copy shader could not be followed: %s",
                      data.failure.c_str());
    } else if (!data.so_failure.empty()) {
        std::snprintf(reason, sizeof(reason), "a stream-out store the capture cannot reproduce: %s",
                      data.so_failure.c_str());
    } else if (data.so_stores.empty()) {
        std::snprintf(reason, sizeof(reason), "no stream-out stores found in the copy shader");
    } else if (data.output_vertices == 0) {
        std::snprintf(reason, sizeof(reason), "the copy shader's GSVS ring layout was not found");
    } else if (dropped || outputs.empty()) {
        std::snprintf(reason, sizeof(reason),
                      "%u of %zu stores have no buffer, fall past their vertex stride or run out "
                      "of output slots",
                      dropped, data.so_stores.size());
    }
    const bool usable = reason[0] == '\0';
    verdicts.emplace(key, usable);
    if (!usable) {
        std::printf("Stream-out: copy shader %016llx refused (its draws are skipped as before): "
                    "%s\n",
                    static_cast<unsigned long long>(params.hash), reason);
    } else {
        std::printf("Stream-out: copy shader %016llx: %zu stores through %zu V#s, %u exported "
                    "parameters; capturing",
                    static_cast<unsigned long long>(params.hash), data.so_stores.size(),
                    data.so_sharps.size(), data.num_attrs);
        for (const auto& output : outputs) {
            std::printf(" [param %u buffer %u +%u x%u]", output.param, output.buffer,
                        output.byte_offset, output.num_components);
        }
        std::printf("\n");
    }
    std::fflush(stdout);
    return usable;
}

void XfbDraw::Begin(vk::CommandBuffer cmdbuf) const {
    for (u32 b = 0; b < num_counters; ++b) {
        if (buffers[b]) {
            cmdbuf.bindTransformFeedbackBuffersEXT(b, 1, &buffers[b], &offsets[b], &sizes[b]);
        }
    }
    cmdbuf.beginTransformFeedbackEXT(0, num_counters, counters.data(), counter_offsets.data());
}

void XfbDraw::End(vk::CommandBuffer cmdbuf) const {
    cmdbuf.endTransformFeedbackEXT(0, num_counters, counters.data(), counter_offsets.data());
}

StreamOut::StreamOut(const Instance& instance_, Runtime& runtime_,
                     VideoCore::BufferCache& buffer_cache_,
                     VideoCore::TextureCache& texture_cache_, Core::MemoryManager* memory_)
    : instance{instance_}, runtime{runtime_}, buffer_cache{buffer_cache_},
      texture_cache{texture_cache_}, memory{memory_} {
    counters = std::make_unique<VideoCore::Buffer>(
        instance, 0, CounterBytes, VideoCore::MemoryType::DeviceLocal, "Stream-out offsets",
        vk::BufferUsageFlagBits::eTransformFeedbackCounterBufferEXT);
    // Offsets start at the beginning of each buffer, as after a GPU reset.
    runtime.FillBuffer(counters.get(), 0, CounterBytes, 0);
}

StreamOut::~StreamOut() = default;

const StreamOut::Layout& StreamOut::LayoutFor(const AmdGpu::Regs& regs) {
    const auto params = AmdGpu::GetParams(regs.vs_program);
    auto [it, inserted] = layouts.try_emplace(params.hash);
    auto& layout = it->second;
    if (inserted) {
        layout.data = Shader::ParseCopyShader(params.code);
    }
    const auto [mask, strides] = BuffersOf(regs);
    if (inserted || mask != layout.buffer_mask || strides != layout.strides) {
        layout.buffer_mask = mask;
        layout.strides = strides;
        layout.outputs.clear();
        if (layout.data.failure.empty() && layout.data.so_failure.empty()) {
            layout.outputs =
                Shader::BuildXfbOutputs(layout.data, mask, strides, layout.data.num_attrs);
        }
    }
    return layout;
}

XfbDraw StreamOut::Prepare(const AmdGpu::Regs& regs, bool& needs_barrier) {
    XfbDraw draw{};
    num_written = 0;
    const auto stages = regs.stage_enable.raw;
    if ((stages != AmdGpu::ShaderStageEnable::VgtStages::EsGs &&
         stages != AmdGpu::ShaderStageEnable::VgtStages::LsHsEsGs) ||
        !instance.IsTransformFeedbackSupported() || !StreamOutHandled(regs)) {
        return draw;
    }
    const auto& layout = LayoutFor(regs);
    if (layout.outputs.empty()) {
        return draw;
    }
    const auto buffers = Shader::MapStreamOutBuffers(layout.data, layout.buffer_mask);
    std::array<std::optional<u64>, 4> bases{};
    std::array<u64, 4> sizes{};
    bool complete = true;
    for (u32 i = 0; i < layout.data.so_sharps.size() && i < buffers.size(); ++i) {
        const s32 b = buffers[i];
        const bool used = b >= 0 && std::ranges::any_of(layout.outputs, [&](const auto& output) {
                              return output.buffer == u32(b);
                          });
        if (!used) {
            continue;
        }
        const auto sharp = ResolveSharp(layout.data.so_sharps[i], regs.vs_program, memory);
        const VAddr base = sharp ? VAddr(sharp->base_address) : 0;
        // The VGT stops at BUFFER_SIZE (dwords from the V# base); else the V# size.
        u64 size = regs.vgt_strmout_buffers[b].size ? u64(regs.vgt_strmout_buffers[b].size) * 4
                                                    : (sharp ? sharp->GetSize() : 0);
        size = base ? memory->ClampRangeSize(base, size) & ~u64{3} : 0;
        if (!sharp || base == 0 || base % 4 != 0 || size == 0) {
            complete = false;
            continue;
        }
        bases[b] = base;
        sizes[b] = size;
    }
    if (++draws == 1) {
        LogFirstDraw(regs, layout, bases, sizes);
    }
    if (!complete) {
        if (!std::exchange(logged_unresolved, true)) {
            std::printf("Stream-out: a stream-out buffer of the draw was not found (V# not traced "
                        "or empty); such draws render without capturing\n");
            std::fflush(stdout);
        }
        return draw;
    }
    for (u32 b = 0; b < 4; ++b) {
        if (!bases[b]) {
            continue;
        }
        VideoCore::written_binding_source = {"stream-out", 0};
        const auto [buffer, offset] = buffer_cache.ObtainBuffer(*bases[b], u32(sizes[b]), true);
        texture_cache.InvalidateMemoryFromGPU(*bases[b], sizes[b]);
        needs_barrier |= runtime.IsBufferAccessed(buffer, offset, sizes[b], true);
        draw.buffers[b] = buffer->Handle();
        draw.offsets[b] = offset;
        draw.sizes[b] = sizes[b];
        draw.counters[b] = counters->Handle();
        draw.counter_offsets[b] = b * sizeof(u32);
        draw.num_counters = b + 1;
        written[num_written++] = {buffer, offset, sizes[b]};
    }
    needs_barrier |= runtime.IsBufferAccessed(counters.get(), 0, CounterBytes, true);
    draw.active = draw.num_counters != 0;
    return draw;
}

void StreamOut::Finish(const XfbDraw& draw) {
    if (!draw.active) {
        return;
    }
    for (u32 i = 0; i < num_written; ++i) {
        runtime.AccessBuffer(written[i].buffer, written[i].offset, written[i].size,
                             vk::PipelineStageFlagBits2::eTransformFeedbackEXT,
                             vk::AccessFlagBits2::eTransformFeedbackWriteEXT);
    }
    num_written = 0;
    // Read at the start of the capture, written at its end. MEMORY_WRITE: the barrier tracker
    // keeps only the write bits it knows, and the counter write is not one of them.
    runtime.AccessBuffer(counters.get(), 0, CounterBytes,
                         vk::PipelineStageFlagBits2::eTransformFeedbackEXT |
                             vk::PipelineStageFlagBits2::eDrawIndirect,
                         vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite);
}

void StreamOut::BufferUpdate(const AmdGpu::PM4CmdStrmoutBufferUpdate& packet) {
    const u32 b = packet.buffer_select.Value();
    const u64 counter_offset = b * sizeof(u32);
    const auto source = packet.source_select.Value();
    if (++updates <= 8) {
        std::printf("Stream-out: buffer update #%llu: buffer %u, source %u (0 packet, 1 filled "
                    "size, 2 memory, 3 none), value %#x, %s, save filled size %u\n",
                    static_cast<unsigned long long>(updates), b, u32(source), packet.buffer_offset,
                    packet.data_type.Value() ? "bytes" : "dwords", packet.update_memory.Value());
        std::fflush(stdout);
    }
    if (!instance.IsTransformFeedbackSupported()) {
        return;
    }
    switch (source) {
    case AmdGpu::SourceSelect::BufferOffset: {
        // The packet counts dwords unless DATA_TYPE says bytes; Vulkan's counters count bytes.
        const u32 bytes = packet.data_type.Value() ? packet.buffer_offset : packet.buffer_offset * 4;
        runtime.InlineData(counters.get(), counter_offset, bytes);
        break;
    }
    case AmdGpu::SourceSelect::VgtStrmoutBufferFilledSize:
        // The offset after the last capture: what the counter already holds.
        break;
    case AmdGpu::SourceSelect::SrcAddress: {
        // A filled size saved earlier (bytes, as saved below).
        const VAddr address = packet.SrcAddress<VAddr>();
        if (address && memory->ClampRangeSize(address, 4) == 4) {
            const auto [buffer, offset] = buffer_cache.ObtainBuffer(address, 4, false);
            const vk::BufferCopy copy{offset, counter_offset, 4};
            runtime.CopyBuffer(buffer, counters.get(), {&copy, 1});
        }
        break;
    }
    case AmdGpu::SourceSelect::None:
        break;
    }
    if (packet.update_memory.Value()) {
        // The filled size in bytes (what radeonsi's DrawTransformFeedback reads back).
        const VAddr address = packet.DstAddress<VAddr>();
        if (address && memory->ClampRangeSize(address, 4) == 4) {
            VideoCore::written_binding_source = {"stream-out size", 0};
            const auto [buffer, offset] = buffer_cache.ObtainBuffer(address, 4, true);
            const vk::BufferCopy copy{counter_offset, offset, 4};
            runtime.CopyBuffer(counters.get(), buffer, {&copy, 1});
        }
    }
}

void StreamOut::LogFirstDraw(const AmdGpu::Regs& regs, const Layout& layout,
                             const std::array<std::optional<u64>, 4>& bases,
                             const std::array<u64, 4>& sizes) {
    std::printf("Stream-out: first draw. VGT_STRMOUT_CONFIG %#x, BUFFER_CONFIG %#x, GS out %u "
                "(0 points, 1 lines, 2 triangles), input primitive %u, %u indices, %u instances, "
                "rasterizer kill %u\n",
                regs.vgt_strmout_config.raw,
                *reinterpret_cast<const u32*>(&regs.vgt_strmout_buffer_config),
                u32(regs.vgt_gs_out_prim_type.GetPrimitiveType(0)), u32(regs.primitive_type),
                regs.num_indices, regs.num_instances.NumInstances(),
                u32(regs.clipper_control.dx_rasterization_kill));
    for (u32 b = 0; b < 4; ++b) {
        const auto& hw = regs.vgt_strmout_buffers[b];
        if (!((layout.buffer_mask >> b) & 1)) {
            continue;
        }
        std::printf("Stream-out:   buffer %u: VGT size %u dwords, stride %u dwords, offset reg %u; "
                    "bound %#llx, %llu bytes\n",
                    b, hw.size, hw.vtx_stride, hw.offset,
                    static_cast<unsigned long long>(bases[b].value_or(0)),
                    static_cast<unsigned long long>(sizes[b]));
    }
    for (u32 i = 0; i < layout.data.so_sharps.size(); ++i) {
        const auto& sharp = layout.data.so_sharps[i];
        const auto vsharp = ResolveSharp(sharp, regs.vs_program, memory);
        std::printf("Stream-out:   V# %u: user data %u, %u hops, offsets %u/%u -> %s base %#llx "
                    "stride %u records %u dfmt %u nfmt %u\n",
                    i, sharp.reg, sharp.hops, sharp.offsets[0], sharp.offsets[1],
                    vsharp ? "found" : "NOT FOUND",
                    static_cast<unsigned long long>(vsharp ? u64(vsharp->base_address) : 0),
                    vsharp ? u32(vsharp->GetStride()) : 0u, vsharp ? u32(vsharp->num_records) : 0u,
                    vsharp ? u32(vsharp->data_format) : 0u, vsharp ? u32(vsharp->num_format) : 0u);
    }
    std::printf("Stream-out:   copy shader: %zu stores, %u exported parameters, %zu outputs:",
                layout.data.so_stores.size(), layout.data.num_attrs, layout.outputs.size());
    for (const auto& output : layout.outputs) {
        std::printf(" [param %u buffer %u +%u x%u ring", output.param, output.buffer,
                    output.byte_offset, output.num_components);
        for (u32 c = 0; c < output.num_components; ++c) {
            std::printf(" %#x", output.ring_offsets[c]);
        }
        std::printf("]");
    }
    std::printf("\n");
    const auto& cb = regs.color_buffers[0];
    const auto& db = regs.depth_buffer;
    std::printf("Stream-out:   color target 0 %#llx %ux%u, %u slices, data format %u; depth %#llx\n",
                static_cast<unsigned long long>(cb ? cb.Address() : 0), cb ? cb.Pitch() : 0u,
                cb ? cb.Height() : 0u, cb ? cb.NumSlices() : 0u,
                cb ? static_cast<unsigned>(cb.GetDataFmt()) : 0u,
                static_cast<unsigned long long>(db.DepthValid() ? db.DepthAddress() : 0));
    std::fflush(stdout); // the programs were printed with the pipeline (StreamOutPipelineUsable)
}

} // namespace Vulkan
