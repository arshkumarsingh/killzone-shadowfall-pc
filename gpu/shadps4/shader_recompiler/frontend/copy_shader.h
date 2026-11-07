// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <map>
#include <span>
#include <vector>

#include "common/types.h"
#include "shader_recompiler/ir/attribute.h"

namespace Shader {

/// gow3 (stream-out): where the copy shader takes a stream-out buffer's V# from. GCN stream-out
/// is plain buffer stores in the GS copy shader at offsets the VGT supplies, so the V# is
/// whatever the shader loads: four user data registers starting at `reg` (`hops` 0), or four
/// dwords read through the 64-bit pointer in user data `reg`/`reg + 1` (`hops` 1: at dword
/// `offsets[0]`), or through a pointer read that way (`hops` 2: pointer at `offsets[0]`, V# at
/// `offsets[1]` from it).
struct StreamOutSharp {
    u32 reg{};
    u32 hops{};
    std::array<u32, 2> offsets{};

    auto operator<=>(const StreamOutSharp&) const = default;
};

/// gow3 (stream-out): one dword the copy shader stores to a stream-out buffer.
struct StreamOutStore {
    u32 ring_offset; ///< GSVS ring dword the value was loaded from (attr_map key space)
    u32 sharp;       ///< index into CopyShaderData::so_sharps
    u32 byte_offset; ///< offset inside the vertex
};

/// gow3 (stream-out): a transform feedback output of the geometry shader: up to four
/// consecutive dwords of one buffer, written to output parameter `param`.
struct XfbOutput {
    u32 param;
    u32 buffer; ///< hardware stream-out buffer (VGT_STRMOUT_*_n), also the XfbBuffer index
    u32 byte_offset;
    u32 num_components;
    std::array<u32, 4> ring_offsets;
};

struct CopyShaderData {
    std::map<u32, std::pair<Shader::IR::Attribute, u32>> attr_map;
    u32 num_attrs{0};
    u32 output_vertices{0};
    u32 num_comps{0};
    /// gow3 (stream-out): the copy shader's stores and the V#s they use (in table order).
    std::vector<StreamOutStore> so_stores;
    std::vector<StreamOutSharp> so_sharps;
    /// gow3 (stream-out): filled by the geometry shader translation (BuildXfbOutputs).
    std::vector<XfbOutput> xfb_outputs;
};

CopyShaderData ParseCopyShader(std::span<const u32> code);

/// gow3 (stream-out): the hardware buffer (0-3) each V# of `data` writes, -1 for none. The V#s
/// in table order take the buffers stream 0 writes (`buffer_mask`, VGT_STRMOUT_BUFFER_CONFIG)
/// from the lowest; the shader (XfbBuffer) and the draw (bindings) both map with this.
std::array<s32, 4> MapStreamOutBuffers(const CopyShaderData& data, u32 buffer_mask);

/// gow3 (stream-out): groups the stores into transform feedback outputs, starting at output
/// parameter `first_param`. `strides` are the vertex strides in dwords (VGT_STRMOUT_VTX_STRIDE).
/// Stores past a stride or past the last parameter are left out (and reported once).
std::vector<XfbOutput> BuildXfbOutputs(const CopyShaderData& data, u32 buffer_mask,
                                       const std::array<u32, 4>& strides, u32 first_param);

} // namespace Shader
