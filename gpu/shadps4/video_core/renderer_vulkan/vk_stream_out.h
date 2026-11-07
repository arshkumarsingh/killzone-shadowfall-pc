// SPDX-License-Identifier: GPL-2.0-or-later
// gow3 (Killzone): GCN stream-out through VK_EXT_transform_feedback.
//
// On GCN the VGT runs the GS copy shader once per emitted vertex of each primitive (strips
// split into lists) and hands it the stream-out state: buffer offsets (VGT_STRMOUT_BUFFER_OFFSET,
// dwords) and the vertex's write index. The copy shader stores what it read from the GSVS ring
// at offset + index * stride, through V#s it loads itself. The VGT checks each primitive against
// VGT_STRMOUT_BUFFER_SIZE and advances the offsets; IT_STRMOUT_BUFFER_UPDATE sets them and saves
// the filled size to memory.
//
// Here the geometry shader writes the streamed dwords to extra outputs decorated for transform
// feedback (copy_shader.cpp: BuildXfbOutputs), the draw captures them into the guest buffers
// (bound in the buffer cache's arenas at the V# base), and four counters in a small buffer play
// the offsets: Vulkan keeps the byte offset of the next vertex there, as the VGT keeps its
// offset, and a primitive that does not fit is dropped as on the VGT. GOW3_STREAMOUT=0: off
// (such draws are skipped as before and the device is made without the extension).
#pragma once

#include <array>
#include <cstdlib>
#include <memory>
#include <optional>
#include <unordered_map>
#include <vector>

#include "common/types.h"
#include "shader_recompiler/frontend/copy_shader.h"
#include "video_core/amdgpu/regs.h"
#include "video_core/renderer_vulkan/vk_common.h"

namespace AmdGpu {
struct PM4CmdStrmoutBufferUpdate;
}

namespace Core {
class MemoryManager;
}

namespace VideoCore {
class BufferCache;
class TextureCache;
struct Buffer;
} // namespace VideoCore

namespace Vulkan {

class Instance;
class Runtime;
class Scheduler;

/// GOW3_STREAMOUT, read once: default on, 0 restores the old skip.
inline bool StreamOutWanted() {
    static const bool wanted = [] {
        const char* env = std::getenv("GOW3_STREAMOUT");
        return !(env && env[0] == '0');
    }();
    return wanted;
}

/// The stream-out configurations emulated: stream 0 only, rasterizing stream 0 (Killzone's one
/// stream-out pipeline: VGT_STRMOUT_CONFIG 0x1). Others are skipped as before.
inline bool StreamOutHandled(const AmdGpu::Regs& regs) {
    const auto& config = regs.vgt_strmout_config;
    return config.streamout_0_en && !config.streamout_1_en && !config.streamout_2_en &&
           !config.streamout_3_en && config.rast_stream == 0 && !config.use_rast_stream_mask;
}

/// Whether a stream-out pipeline with `regs` (StreamOutHandled) can be emulated: its copy shader
/// parsed, every stream-out store reproducible, something to capture. A new pipeline (by ES, GS
/// and copy shader, buffers and strides) first prints its three programs, then the verdict (one
/// line saying why when refused). Any thread. `cached_only` (draw preparation workers): decide
/// from earlier verdicts only, std::nullopt when there is none.
std::optional<bool> StreamOutPipelineUsable(const AmdGpu::Regs& regs, bool cached_only);

/// What a stream-out draw binds; copied into the recorded draw command.
struct XfbDraw {
    bool active = false;
    u32 num_counters = 0; ///< counters 0 .. num_counters - 1 (a null buffer where unused)
    std::array<vk::Buffer, 4> buffers{};
    std::array<vk::DeviceSize, 4> offsets{};
    std::array<vk::DeviceSize, 4> sizes{};
    std::array<vk::Buffer, 4> counters{};
    std::array<vk::DeviceSize, 4> counter_offsets{};

    /// Inside the render pass, around the draw: binds the buffers and starts the capture at the
    /// counters' offsets; End saves the new offsets.
    void Begin(vk::CommandBuffer cmdbuf) const;
    void End(vk::CommandBuffer cmdbuf) const;
};

class StreamOut {
public:
    explicit StreamOut(const Instance& instance, Runtime& runtime,
                       VideoCore::BufferCache& buffer_cache,
                       VideoCore::TextureCache& texture_cache, Core::MemoryManager* memory);
    ~StreamOut();

    /// A geometry draw with `regs`: the guest buffers it streams out to, obtained as written
    /// (before the render pass begins). Inactive when the draw does not stream out or a V# is
    /// not found (the draw then renders without capturing). Sets `needs_barrier` on hazards.
    XfbDraw Prepare(const AmdGpu::Regs& regs, bool& needs_barrier);

    /// After the draw is recorded: its writes, for the barriers of later readers.
    void Finish(const XfbDraw& draw);

    /// IT_STRMOUT_BUFFER_UPDATE: sets a buffer's offset (from the packet, or memory) and saves
    /// its filled size to memory.
    void BufferUpdate(const AmdGpu::PM4CmdStrmoutBufferUpdate& packet);

private:
    struct Layout {
        Shader::CopyShaderData data;
        u32 buffer_mask = ~0u;
        std::array<u32, 4> strides{};
        std::vector<Shader::XfbOutput> outputs;
    };

    const Layout& LayoutFor(const AmdGpu::Regs& regs);
    void LogFirstDraw(const AmdGpu::Regs& regs, const Layout& layout,
                      const std::array<std::optional<u64>, 4>& bases,
                      const std::array<u64, 4>& sizes);

    const Instance& instance;
    Runtime& runtime;
    VideoCore::BufferCache& buffer_cache;
    VideoCore::TextureCache& texture_cache;
    Core::MemoryManager* memory;
    std::unique_ptr<VideoCore::Buffer> counters; ///< four u32 byte offsets (Vulkan's counters)
    std::unordered_map<u64, Layout> layouts;     ///< by copy shader hash
    struct Written {
        const VideoCore::Buffer* buffer;
        u64 offset;
        u64 size;
    };
    std::array<Written, 4> written{};
    u32 num_written = 0;
    u64 draws = 0;
    u64 updates = 0;
    bool logged_unresolved = false;
};

} // namespace Vulkan
