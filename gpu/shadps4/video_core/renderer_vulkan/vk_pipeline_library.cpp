// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string_view>
#include <type_traits>
#include <boost/container/static_vector.hpp>
#include <xxhash.h>

#include "common/elf_info.h"
#include "common/logging/log.h"
#include "common/path_util.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_pipeline_library.h"
#include "video_core/renderer_vulkan/vk_pipeline_serialization.h"

namespace Vulkan {

namespace {
using Clock = std::chrono::steady_clock;

u64 NsSince(Clock::time_point start) {
    return u64(std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start).count());
}

void AtomicMax(std::atomic<u64>& target, u64 value) {
    u64 seen = target.load(std::memory_order_relaxed);
    while (value > seen && !target.compare_exchange_weak(seen, value, std::memory_order_relaxed)) {
    }
}

constexpr vk::GraphicsPipelineLibraryFlagsEXT PartFlag(PipelineLibraries::Part part) {
    switch (part) {
    case PipelineLibraries::VertexInput:
        return vk::GraphicsPipelineLibraryFlagBitsEXT::eVertexInputInterface;
    case PipelineLibraries::PreRaster:
        return vk::GraphicsPipelineLibraryFlagBitsEXT::ePreRasterizationShaders;
    case PipelineLibraries::FragmentShader:
        return vk::GraphicsPipelineLibraryFlagBitsEXT::eFragmentShader;
    default:
        return vk::GraphicsPipelineLibraryFlagBitsEXT::eFragmentOutputInterface;
    }
}

constexpr const char* PartName(u32 part) {
    constexpr std::array names{"vertex input", "pre-raster", "fragment", "output"};
    return names[part];
}

bool IsFragmentStage(const vk::PipelineShaderStageCreateInfo& stage) {
    return stage.stage == vk::ShaderStageFlagBits::eFragment;
}

const vk::BaseInStructure* Next(const void* p_next) {
    return static_cast<const vk::BaseInStructure*>(p_next);
}
} // namespace

/// XXH3-128 of the state a library part reads, field by field (struct padding is not hashed).
/// The fields are gathered and hashed once (no streaming state: XXH3_state_t is only complete
/// with XXH_STATIC_LINKING_ONLY).
class PartHasher {
public:
    PartHasher() {
        bytes.reserve(1024);
    }
    template <typename T>
        requires std::is_trivially_copyable_v<T> && (!std::is_pointer_v<T>)
    void Add(const T& value) {
        static_assert(std::has_unique_object_representations_v<T> || std::is_floating_point_v<T>,
                      "hash scalars only");
        Bytes(&value, sizeof(value));
    }
    void Bytes(const void* data, size_t size) {
        const auto* first = static_cast<const u8*>(data);
        bytes.insert(bytes.end(), first, first + size);
    }
    void Flag(VkFlags flags) {
        Add(u32(flags));
    }
    template <typename E>
    void Enum(E value) {
        Add(static_cast<u32>(value));
    }
    void Bool(vk::Bool32 value) {
        Add(u32(value));
    }
    void Handle(u64 value) {
        Add(value);
    }
    PipelineLibraries::Key Finish() const {
        const XXH128_hash_t hash = XXH3_128bits(bytes.data(), bytes.size());
        return {hash.low64, hash.high64};
    }

private:
    std::vector<u8> bytes;
};

namespace {
template <typename F>
bool WalkChain(const void* p_next, F&& visit) {
    for (const auto* s = Next(p_next); s; s = s->pNext) {
        if (!visit(*s)) {
            return false;
        }
    }
    return true;
}

bool HashVertexInput(PartHasher& h, const vk::PipelineVertexInputStateCreateInfo* vi) {
    h.Add(u32(vi != nullptr));
    if (!vi) {
        return true;
    }
    h.Flag(VkFlags(vi->flags));
    h.Add(vi->vertexBindingDescriptionCount);
    for (u32 i = 0; i < vi->vertexBindingDescriptionCount; ++i) {
        const auto& b = vi->pVertexBindingDescriptions[i];
        h.Add(b.binding);
        h.Add(b.stride);
        h.Enum(b.inputRate);
    }
    h.Add(vi->vertexAttributeDescriptionCount);
    for (u32 i = 0; i < vi->vertexAttributeDescriptionCount; ++i) {
        const auto& a = vi->pVertexAttributeDescriptions[i];
        h.Add(a.location);
        h.Add(a.binding);
        h.Enum(a.format);
        h.Add(a.offset);
    }
    return WalkChain(vi->pNext, [&](const vk::BaseInStructure& s) {
        if (s.sType != vk::StructureType::ePipelineVertexInputDivisorStateCreateInfo) {
            return false;
        }
        const auto& d = reinterpret_cast<const vk::PipelineVertexInputDivisorStateCreateInfo&>(s);
        h.Add(d.vertexBindingDivisorCount);
        for (u32 i = 0; i < d.vertexBindingDivisorCount; ++i) {
            h.Add(d.pVertexBindingDivisors[i].binding);
            h.Add(d.pVertexBindingDivisors[i].divisor);
        }
        return true;
    });
}

bool HashInputAssembly(PartHasher& h, const vk::PipelineInputAssemblyStateCreateInfo* ia) {
    h.Add(u32(ia != nullptr));
    if (!ia) {
        return true;
    }
    h.Flag(VkFlags(ia->flags));
    h.Enum(ia->topology);
    h.Bool(ia->primitiveRestartEnable);
    return ia->pNext == nullptr;
}

bool HashTessellation(PartHasher& h, const vk::PipelineTessellationStateCreateInfo* ts) {
    h.Add(u32(ts != nullptr));
    if (!ts) {
        return true;
    }
    h.Flag(VkFlags(ts->flags));
    h.Add(ts->patchControlPoints);
    return ts->pNext == nullptr;
}

bool HashViewport(PartHasher& h, const vk::PipelineViewportStateCreateInfo* vp) {
    h.Add(u32(vp != nullptr));
    if (!vp) {
        return true;
    }
    h.Flag(VkFlags(vp->flags));
    h.Add(vp->viewportCount);
    h.Add(vp->scissorCount);
    if (vp->pViewports) {
        for (u32 i = 0; i < vp->viewportCount; ++i) {
            const auto& v = vp->pViewports[i];
            h.Add(v.x);
            h.Add(v.y);
            h.Add(v.width);
            h.Add(v.height);
            h.Add(v.minDepth);
            h.Add(v.maxDepth);
        }
    }
    if (vp->pScissors) {
        for (u32 i = 0; i < vp->scissorCount; ++i) {
            const auto& s = vp->pScissors[i];
            h.Add(s.offset.x);
            h.Add(s.offset.y);
            h.Add(s.extent.width);
            h.Add(s.extent.height);
        }
    }
    return WalkChain(vp->pNext, [&](const vk::BaseInStructure& s) {
        if (s.sType != vk::StructureType::ePipelineViewportDepthClipControlCreateInfoEXT) {
            return false;
        }
        h.Bool(reinterpret_cast<const vk::PipelineViewportDepthClipControlCreateInfoEXT&>(s)
                   .negativeOneToOne);
        return true;
    });
}

bool HashRasterization(PartHasher& h, const vk::PipelineRasterizationStateCreateInfo* rs) {
    h.Add(u32(rs != nullptr));
    if (!rs) {
        return true;
    }
    h.Flag(VkFlags(rs->flags));
    h.Bool(rs->depthClampEnable);
    h.Bool(rs->rasterizerDiscardEnable);
    h.Enum(rs->polygonMode);
    h.Flag(VkFlags(rs->cullMode));
    h.Enum(rs->frontFace);
    h.Bool(rs->depthBiasEnable);
    h.Add(rs->depthBiasConstantFactor);
    h.Add(rs->depthBiasClamp);
    h.Add(rs->depthBiasSlopeFactor);
    h.Add(rs->lineWidth);
    return WalkChain(rs->pNext, [&](const vk::BaseInStructure& s) {
        if (s.sType == vk::StructureType::ePipelineRasterizationProvokingVertexStateCreateInfoEXT) {
            h.Add(u32(1));
            h.Enum(reinterpret_cast<
                       const vk::PipelineRasterizationProvokingVertexStateCreateInfoEXT&>(s)
                       .provokingVertexMode);
            return true;
        }
        if (s.sType == vk::StructureType::ePipelineRasterizationDepthClipStateCreateInfoEXT) {
            const auto& clip =
                reinterpret_cast<const vk::PipelineRasterizationDepthClipStateCreateInfoEXT&>(s);
            h.Add(u32(2));
            h.Flag(VkFlags(clip.flags));
            h.Bool(clip.depthClipEnable);
            return true;
        }
        return false; // e.g. stream-out rasterization state: not linked
    });
}

bool HashMultisample(PartHasher& h, const vk::PipelineMultisampleStateCreateInfo* ms) {
    h.Add(u32(ms != nullptr));
    if (!ms) {
        return true;
    }
    h.Flag(VkFlags(ms->flags));
    h.Flag(VkFlags(ms->rasterizationSamples));
    h.Bool(ms->sampleShadingEnable);
    h.Add(ms->minSampleShading);
    h.Add(u32(ms->pSampleMask != nullptr));
    if (ms->pSampleMask) {
        const u32 words = (u32(ms->rasterizationSamples) + 31) / 32;
        h.Bytes(ms->pSampleMask, words * sizeof(u32));
    }
    h.Bool(ms->alphaToCoverageEnable);
    h.Bool(ms->alphaToOneEnable);
    return ms->pNext == nullptr;
}

void HashStencil(PartHasher& h, const vk::StencilOpState& s) {
    h.Enum(s.failOp);
    h.Enum(s.passOp);
    h.Enum(s.depthFailOp);
    h.Enum(s.compareOp);
    h.Add(s.compareMask);
    h.Add(s.writeMask);
    h.Add(s.reference);
}

bool HashDepthStencil(PartHasher& h, const vk::PipelineDepthStencilStateCreateInfo* ds) {
    h.Add(u32(ds != nullptr));
    if (!ds) {
        return true;
    }
    h.Flag(VkFlags(ds->flags));
    h.Bool(ds->depthTestEnable);
    h.Bool(ds->depthWriteEnable);
    h.Enum(ds->depthCompareOp);
    h.Bool(ds->depthBoundsTestEnable);
    h.Bool(ds->stencilTestEnable);
    HashStencil(h, ds->front);
    HashStencil(h, ds->back);
    h.Add(ds->minDepthBounds);
    h.Add(ds->maxDepthBounds);
    return ds->pNext == nullptr;
}

bool HashColorBlend(PartHasher& h, const vk::PipelineColorBlendStateCreateInfo* cb) {
    h.Add(u32(cb != nullptr));
    if (!cb) {
        return true;
    }
    h.Flag(VkFlags(cb->flags));
    h.Bool(cb->logicOpEnable);
    h.Enum(cb->logicOp);
    h.Add(cb->attachmentCount);
    for (u32 i = 0; i < cb->attachmentCount; ++i) {
        const auto& a = cb->pAttachments[i];
        h.Bool(a.blendEnable);
        h.Enum(a.srcColorBlendFactor);
        h.Enum(a.dstColorBlendFactor);
        h.Enum(a.colorBlendOp);
        h.Enum(a.srcAlphaBlendFactor);
        h.Enum(a.dstAlphaBlendFactor);
        h.Enum(a.alphaBlendOp);
        h.Flag(VkFlags(a.colorWriteMask));
    }
    for (const float c : cb->blendConstants) {
        h.Add(c);
    }
    return cb->pNext == nullptr;
}

bool HashDynamic(PartHasher& h, const vk::PipelineDynamicStateCreateInfo* dy) {
    h.Add(u32(dy != nullptr));
    if (!dy) {
        return true;
    }
    h.Flag(VkFlags(dy->flags));
    h.Add(dy->dynamicStateCount);
    for (u32 i = 0; i < dy->dynamicStateCount; ++i) {
        h.Enum(dy->pDynamicStates[i]);
    }
    return dy->pNext == nullptr;
}

/// The create info's own chain: dynamic rendering formats and the AMD sample counts. Anything
/// else (a structure the monolithic path gains later) makes the pipeline unsupported.
bool HashRendering(PartHasher& h, const vk::GraphicsPipelineCreateInfo& info) {
    return WalkChain(info.pNext, [&](const vk::BaseInStructure& s) {
        if (s.sType == vk::StructureType::ePipelineRenderingCreateInfo) {
            const auto& r = reinterpret_cast<const vk::PipelineRenderingCreateInfo&>(s);
            h.Add(u32(1));
            h.Add(r.viewMask);
            h.Add(r.colorAttachmentCount);
            for (u32 i = 0; i < r.colorAttachmentCount; ++i) {
                h.Enum(r.pColorAttachmentFormats[i]);
            }
            h.Enum(r.depthAttachmentFormat);
            h.Enum(r.stencilAttachmentFormat);
            return true;
        }
        if (s.sType == vk::StructureType::eAttachmentSampleCountInfoAMD) {
            const auto& a = reinterpret_cast<const vk::AttachmentSampleCountInfoAMD&>(s);
            h.Add(u32(2));
            h.Add(a.colorAttachmentCount);
            for (u32 i = 0; i < a.colorAttachmentCount && a.pColorAttachmentSamples; ++i) {
                h.Flag(VkFlags(a.pColorAttachmentSamples[i]));
            }
            h.Flag(VkFlags(a.depthStencilAttachmentSamples));
            return true;
        }
        return false;
    });
}
} // namespace

PipelineLibraries::PipelineLibraries(const Instance& instance_) : instance{instance_} {
    enabled = instance.IsGraphicsPipelineLibraryEnabled();
    if (const char* env = std::getenv("GOW3_GPL_MAX_LIBRARIES"); env && *env) {
        max_libraries = size_t(std::max(0L, std::strtol(env, nullptr, 10)));
    }
    if (enabled) {
        std::printf("GPU: fast-link pipelines (GOW3_GPL) on: a pipeline not built yet is linked at "
                    "once from stage libraries while the optimized one compiles (up to %zu "
                    "libraries)\n",
                    max_libraries);
    } else {
        std::printf("GPU: fast-link pipelines (GOW3_GPL) %s\n",
                    instance.GraphicsPipelineLibraryStatus());
    }
    std::fflush(stdout);
}

PipelineLibraries::~PipelineLibraries() = default;

size_t PipelineLibraries::Count() const {
    std::scoped_lock lock{mutex};
    return total;
}

vk::ResultValue<vk::UniquePipeline> PipelineLibraries::CreateLibrary(
    Part part, const vk::GraphicsPipelineCreateInfo& info, vk::PipelineCache cache) const {
    vk::GraphicsPipelineLibraryCreateInfoEXT library{.flags = PartFlag(part)};
    boost::container::static_vector<vk::PipelineShaderStageCreateInfo, MaxShaderStages> stages;
    // Fragment shader state without the output interface needs a depth/stencil state with
    // dynamic rendering; everything in it is dynamic here (the monolithic path may pass null).
    static constexpr vk::PipelineDepthStencilStateCreateInfo no_depth_stencil{};
    vk::GraphicsPipelineCreateInfo ci{
        .pNext = &library,
        .flags = info.flags | vk::PipelineCreateFlagBits::eLibraryKHR,
        .pDynamicState = info.pDynamicState,
    };
    switch (part) {
    case VertexInput:
        ci.pVertexInputState = info.pVertexInputState;
        ci.pInputAssemblyState = info.pInputAssemblyState;
        break;
    case PreRaster:
        for (u32 i = 0; i < info.stageCount; ++i) {
            if (!IsFragmentStage(info.pStages[i])) {
                stages.push_back(info.pStages[i]);
            }
        }
        ci.stageCount = u32(stages.size());
        ci.pStages = stages.data();
        ci.pTessellationState = info.pTessellationState;
        ci.pViewportState = info.pViewportState;
        ci.pRasterizationState = info.pRasterizationState;
        ci.layout = info.layout;
        break;
    case FragmentShader:
        for (u32 i = 0; i < info.stageCount; ++i) {
            if (IsFragmentStage(info.pStages[i])) {
                stages.push_back(info.pStages[i]);
            }
        }
        ci.stageCount = u32(stages.size());
        ci.pStages = stages.data();
        ci.pMultisampleState = info.pMultisampleState;
        ci.pDepthStencilState =
            info.pDepthStencilState ? info.pDepthStencilState : &no_depth_stencil;
        ci.layout = info.layout;
        break;
    default:
        // Attachment formats and sample counts ride on the create info's own chain.
        library.pNext = const_cast<void*>(info.pNext);
        ci.pColorBlendState = info.pColorBlendState;
        ci.pMultisampleState = info.pMultisampleState;
        break;
    }
    return instance.GetDevice().createGraphicsPipelineUnique(cache, ci);
}

vk::Pipeline PipelineLibraries::Acquire(Part part, const Key& key,
                                        const vk::GraphicsPipelineCreateInfo& info,
                                        vk::PipelineCache cache, bool may_create, bool may_wait) {
    std::unique_lock lock{mutex};
    auto& map = entries[part];
    if (const auto it = map.find(key); it != map.end()) {
        Entry& entry = *it->second;
        if (!entry.done) {
            if (!may_wait) {
                return {};
            }
            cv.wait(lock, [&] { return entry.done; });
        }
        if (entry.pipeline) {
            counters.reused.fetch_add(1, std::memory_order_relaxed);
        }
        return *entry.pipeline;
    }
    if (!may_create || total >= max_libraries) {
        return {};
    }
    Entry& entry = *map.emplace(key, std::make_unique<Entry>()).first->second;
    ++total;
    lock.unlock();

    const auto started = Clock::now();
    auto [result, pipeline] = CreateLibrary(part, info, cache);
    const u64 ns = NsSince(started);

    lock.lock();
    if (result == vk::Result::eSuccess) {
        entry.pipeline = std::move(pipeline);
        counters.compiled[part].fetch_add(1, std::memory_order_relaxed);
        counters.compile_ns[part].fetch_add(ns, std::memory_order_relaxed);
        AtomicMax(counters.compile_worst_ns[part], ns);
        if (part == PreRaster || part == FragmentShader) {
            generation.fetch_add(1, std::memory_order_release);
        }
    } else {
        counters.failed.fetch_add(1, std::memory_order_relaxed);
        LOG_WARNING(Render_Vulkan, "gow3: {} pipeline library failed: {}", PartName(part),
                    vk::to_string(result));
    }
    entry.done = true;
    cv.notify_all();
    return *entry.pipeline;
}

PipelineLinker::PipelineLinker(PipelineLibraries& libraries_, Mode mode_, vk::PipelineCache cache_,
                               const GraphicsPipelineKey& key,
                               std::span<const vk::ShaderModule> modules_,
                               const GraphicsPipeline::SerializationSupport& sdata_)
    : libraries{libraries_}, mode{mode_}, cache{cache_}, sdata{sdata_} {
    std::ranges::copy(key.stage_hashes, stage_hashes.begin());
    std::copy_n(modules_.begin(), std::min<size_t>(modules_.size(), modules.size()),
                modules.begin());
}

PipelineLinker::~PipelineLinker() {
    const auto device = libraries.instance.GetDevice();
    for (const auto module : aux_modules) {
        device.destroyShaderModule(module);
    }
}

bool PipelineLinker::HashModule(PartHasher& h, const vk::PipelineShaderStageCreateInfo& stage) {
    for (u32 i = 0; i < MaxShaderStages; ++i) {
        if (modules[i] && modules[i] == stage.module) {
            h.Add(u32(1));
            h.Add(stage_hashes[i]); // program and permutation
            h.Handle(u64(reinterpret_cast<uintptr_t>(static_cast<VkShaderModule>(stage.module))));
            return true;
        }
    }
    // A stage the constructor generated: identified by its SPIR-V.
    const std::vector<u32>* code = nullptr;
    switch (stage.stage) {
    case vk::ShaderStageFlagBits::eTessellationControl:
        code = &sdata.tcs;
        break;
    case vk::ShaderStageFlagBits::eTessellationEvaluation:
        code = &sdata.tes;
        break;
    case vk::ShaderStageFlagBits::eFragment:
        code = &sdata.fragment;
        break;
    default:
        return false;
    }
    if (code->empty()) {
        return false;
    }
    h.Add(u32(2));
    h.Bytes(code->data(), code->size() * sizeof(u32));
    if (std::ranges::find(aux_modules, stage.module) == aux_modules.end()) {
        aux_modules.push_back(stage.module);
    }
    return true;
}

bool PipelineLinker::HashPart(PipelineLibraries::Part part, const GraphicsPipeline& pipeline,
                              const vk::GraphicsPipelineCreateInfo& info,
                              PipelineLibraries::Key& out) {
    PartHasher h;
    h.Add(u32(part));
    h.Flag(VkFlags(info.flags));
    if (!HashDynamic(h, info.pDynamicState)) {
        return false;
    }
    const auto hash_stages = [&](bool fragment) {
        u32 count = 0;
        for (u32 i = 0; i < info.stageCount; ++i) {
            const auto& stage = info.pStages[i];
            if (IsFragmentStage(stage) != fragment) {
                continue;
            }
            ++count;
            h.Flag(VkFlags(stage.stage));
            h.Flag(VkFlags(stage.flags));
            const std::string_view name{stage.pName ? stage.pName : ""};
            h.Add(u32(name.size()));
            h.Bytes(name.data(), name.size());
            if (stage.pSpecializationInfo || !HashModule(h, stage)) {
                return false;
            }
            const bool chain_ok = WalkChain(stage.pNext, [&](const vk::BaseInStructure& s) {
                if (s.sType != vk::StructureType::ePipelineShaderStageRequiredSubgroupSizeCreateInfo) {
                    return false;
                }
                h.Add(reinterpret_cast<const vk::PipelineShaderStageRequiredSubgroupSizeCreateInfo&>(s)
                          .requiredSubgroupSize);
                return true;
            });
            if (!chain_ok) {
                return false;
            }
        }
        h.Add(count);
        return true;
    };
    const auto hash_layout = [&] {
        const auto bindings = pipeline.DescriptorBindings();
        h.Add(u32(pipeline.PushDescriptors()));
        h.Add(u32(bindings.size()));
        for (const auto& b : bindings) {
            if (b.pImmutableSamplers) {
                return false;
            }
            h.Add(b.binding);
            h.Enum(b.descriptorType);
            h.Add(b.descriptorCount);
            h.Flag(VkFlags(b.stageFlags));
        }
        return true;
    };
    bool ok = false;
    switch (part) {
    case PipelineLibraries::VertexInput:
        ok = HashVertexInput(h, info.pVertexInputState) &&
             HashInputAssembly(h, info.pInputAssemblyState);
        break;
    case PipelineLibraries::PreRaster:
        ok = hash_stages(false) && HashTessellation(h, info.pTessellationState) &&
             HashViewport(h, info.pViewportState) &&
             HashRasterization(h, info.pRasterizationState) && hash_layout();
        break;
    case PipelineLibraries::FragmentShader:
        ok = hash_stages(true) && HashMultisample(h, info.pMultisampleState) &&
             HashDepthStencil(h, info.pDepthStencilState) && hash_layout();
        break;
    default:
        ok = HashColorBlend(h, info.pColorBlendState) &&
             HashMultisample(h, info.pMultisampleState);
        break;
    }
    // The create info's own chain: the formats are only the output's state (the other parts
    // must not depend on them), but an unknown structure there means no part is linked.
    if (part == PipelineLibraries::FragmentOutput) {
        ok = ok && HashRendering(h, info);
    } else {
        PartHasher scratch;
        ok = ok && HashRendering(scratch, info);
    }
    out = h.Finish();
    return ok;
}

bool PipelineLinker::PipelineBinaryOnDisk(const vk::GraphicsPipelineCreateInfo& info) const {
    const auto& instance = libraries.instance;
    const auto& global = instance.GetPipelineBinaryGlobalKey();
    if (!global.keySize) {
        return false;
    }
    const auto started = Clock::now();
    const auto directory = Common::FS::GetUserPath(Common::FS::PathType::CacheDir) /
                           Common::ElfInfo::Instance().GameSerial() / "pipeline-binaries";
    const bool found = Vulkan::PipelineBinaryOnDisk(instance.GetDevice(), info, global, directory);
    libraries.counters.probe_key_ns.fetch_add(NsSince(started), std::memory_order_relaxed);
    libraries.counters.probe_keys.fetch_add(1, std::memory_order_relaxed);
    return found;
}

vk::ResultValue<vk::UniquePipeline> PipelineLinker::Create(const GraphicsPipeline& pipeline,
                                                           const vk::GraphicsPipelineCreateInfo& info,
                                                           bool square) {
    using Libs = PipelineLibraries;
    const auto not_linked = [&](Outcome why) {
        outcome = why;
        return vk::ResultValue<vk::UniquePipeline>{vk::Result::eIncomplete, vk::UniquePipeline{}};
    };
    std::array<Libs::Key, Libs::NumParts> keys;
    for (u32 part = 0; part < Libs::NumParts; ++part) {
        if (!HashPart(Libs::Part(part), pipeline, info, keys[part])) {
            return not_linked(Outcome::Unsupported);
        }
    }
    std::array<vk::Pipeline, Libs::NumParts> parts{};
    if (mode == Mode::Probe) {
        // Shader parts: only when ready. The binary check costs a driver key computation, so
        // it runs only when a link is not possible right now.
        for (const auto part : {Libs::PreRaster, Libs::FragmentShader}) {
            parts[part] = libraries.Acquire(part, keys[part], info, {}, false, false);
        }
        if (!parts[Libs::PreRaster] || !parts[Libs::FragmentShader]) {
            if (check_binary && !square && PipelineBinaryOnDisk(info)) {
                return not_linked(Outcome::BinaryCached);
            }
            return not_linked(Outcome::MissingLibraries);
        }
        // The cheap parts have no shader to compile: made here when missing.
        for (const auto part : {Libs::VertexInput, Libs::FragmentOutput}) {
            parts[part] = libraries.Acquire(part, keys[part], info, {}, true, false);
            if (!parts[part]) {
                return not_linked(Outcome::MissingLibraries);
            }
        }
    } else {
        for (const auto part :
             {Libs::VertexInput, Libs::FragmentOutput, Libs::PreRaster, Libs::FragmentShader}) {
            if (cancel && cancel->load(std::memory_order_relaxed)) {
                return not_linked(Outcome::Cancelled);
            }
            parts[part] = libraries.Acquire(part, keys[part], info, cache, true, true);
            if (!parts[part]) {
                return not_linked(Outcome::Failed);
            }
        }
    }

    const vk::PipelineLibraryCreateInfoKHR link_info{
        .libraryCount = u32(parts.size()),
        .pLibraries = parts.data(),
    };
    // No link-time optimization: that is the fast link (graphicsPipelineLibraryFastLinking).
    const vk::GraphicsPipelineCreateInfo link{
        .pNext = &link_info,
        .flags = info.flags,
        .layout = info.layout,
    };
    const auto started = Clock::now();
    auto result = libraries.instance.GetDevice().createGraphicsPipelineUnique({}, link);
    const u64 ns = NsSince(started);
    auto& counters = libraries.counters;
    if (result.result != vk::Result::eSuccess) {
        counters.failed.fetch_add(1, std::memory_order_relaxed);
        LOG_WARNING(Render_Vulkan, "gow3: pipeline fast link failed: {}",
                    vk::to_string(result.result));
        return not_linked(Outcome::Failed);
    }
    counters.links.fetch_add(1, std::memory_order_relaxed);
    counters.link_ns.fetch_add(ns, std::memory_order_relaxed);
    AtomicMax(counters.link_worst_ns, ns);
    outcome = Outcome::Linked;
    return result;
}

void FastLinkState::Report() {
    const auto now = std::chrono::steady_clock::now();
    if (now - last_report < std::chrono::seconds(5)) {
        return;
    }
    auto& c = libraries.counters;
    std::array<u32, PipelineLibraries::NumParts> compiled{};
    std::array<u64, PipelineLibraries::NumParts> compile_ns{};
    u32 compiled_any = 0;
    for (u32 part = 0; part < PipelineLibraries::NumParts; ++part) {
        const u32 total = c.compiled[part].load(std::memory_order_relaxed);
        const u64 ns = c.compile_ns[part].load(std::memory_order_relaxed);
        compiled[part] = total - compiled_seen[part];
        compile_ns[part] = ns - compile_ns_seen[part];
        compiled_seen[part] = total;
        compile_ns_seen[part] = ns;
        compiled_any += compiled[part];
    }
    const u32 reused_total = c.reused.load(std::memory_order_relaxed);
    const u32 reused = reused_total - reused_seen;
    reused_seen = reused_total;
    const u32 links_total = c.links.load(std::memory_order_relaxed);
    const u64 link_ns_total = c.link_ns.load(std::memory_order_relaxed);
    const u32 links = links_total - links_seen;
    const u64 link_ns = link_ns_total - link_ns_seen;
    links_seen = links_total;
    link_ns_seen = link_ns_total;
    const u32 keys_total = c.probe_keys.load(std::memory_order_relaxed);
    const u64 key_ns_total = c.probe_key_ns.load(std::memory_order_relaxed);
    const u32 binary_checks = keys_total - probe_keys_seen;
    const u64 binary_check_ns = key_ns_total - probe_key_ns_seen;
    probe_keys_seen = keys_total;
    probe_key_ns_seen = key_ns_total;

    const bool fast_activity = linked_now || linked_async || replaced || optimized_first ||
                               binary_first || unsupported || failed || compiled_any;
    const bool timing_activity = ready_shader.count || ready_fast.count ||
                                 ready_optimized.count || pending_shader || pending_pipeline;
    if (!fast_activity && !timing_activity) {
        return;
    }
    last_report = now;
    const auto avg = [](u64 ns, u32 count) { return count ? double(ns) / 1e6 / count : 0.0; };
    if (libraries.Enabled() && fast_activity) {
        std::printf(
            "Shaders: fast link: %u linked at once, %u after compiling libraries, %u swapped for "
            "the optimized pipeline, %u optimized first; %u restored from binaries instead (%u "
            "binary checks, avg %.2f ms), %u not linkable, %u failed; libraries compiled: %u "
            "pre-raster (avg %.1f ms), %u fragment (avg %.1f ms), %u vertex input, %u output, %u "
            "reused; %u links (avg %.2f ms, worst %.2f ms); %zu libraries, %zu pipelines kept\n",
            linked_now, linked_async, replaced, optimized_first, binary_first, binary_checks,
            avg(binary_check_ns, binary_checks), unsupported, failed,
            compiled[PipelineLibraries::PreRaster],
            avg(compile_ns[PipelineLibraries::PreRaster], compiled[PipelineLibraries::PreRaster]),
            compiled[PipelineLibraries::FragmentShader],
            avg(compile_ns[PipelineLibraries::FragmentShader],
                compiled[PipelineLibraries::FragmentShader]),
            compiled[PipelineLibraries::VertexInput], compiled[PipelineLibraries::FragmentOutput],
            reused, links, avg(link_ns, links), c.link_worst_ns.load() / 1e6, libraries.Count(),
            kept.size());
    }
    if (timing_activity) {
        const auto line = [](const Timing& t) {
            return t.count ? t.total_ms / t.count : 0.0;
        };
        std::printf("Shaders: ready after (avg/worst ms, count): shaders %.0f/%.0f (%u), "
                    "fast-linked pipelines %.0f/%.0f (%u), optimized pipelines %.0f/%.0f (%u); "
                    "lookups waiting on a shader %u, on a pipeline %u\n",
                    line(ready_shader), ready_shader.worst_ms, ready_shader.count, line(ready_fast),
                    ready_fast.worst_ms, ready_fast.count, line(ready_optimized),
                    ready_optimized.worst_ms, ready_optimized.count, pending_shader,
                    pending_pipeline);
    }
    std::fflush(stdout);
    linked_now = linked_async = replaced = optimized_first = 0;
    binary_first = unsupported = failed = 0;
    pending_shader = pending_pipeline = 0;
    ready_shader = ready_fast = ready_optimized = {};
}

} // namespace Vulkan
