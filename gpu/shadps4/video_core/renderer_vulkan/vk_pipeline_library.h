// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <span>
#include <unordered_map>
#include <vector>
#include <tsl/robin_map.h>
#include <tsl/robin_set.h>

#include "video_core/renderer_vulkan/vk_common.h"
#include "video_core/renderer_vulkan/vk_graphics_pipeline.h"

namespace Vulkan {

class Instance;
class PartHasher;
struct AsyncGraphicsCompilation;

/// gow3 (Killzone): GOW3_GPL fast link, after DXVK. A draw whose pipeline is not built yet gets
/// one linked at once from four VK_EXT_graphics_pipeline_library parts, without link-time
/// optimization; the optimized (monolithic) pipeline still compiles in the background and
/// replaces it (PipelineCache::PublishGraphics).
///
/// The parts are cut out of the very create info GraphicsPipeline's constructor builds for the
/// monolithic pipeline (PipelineLinker::Create), so a linked pipeline has the same state as the
/// optimized one by construction:
///   vertex input     vertex input state, input assembly                      (no compile)
///   pre-rasterization vertex/tessellation/geometry stages, viewport, raster, layout
///   fragment shader  fragment stage, multisample, depth/stencil, layout
///   fragment output  color blend, multisample, attachment formats and samples (no compile)
/// Every part gets the pipeline's whole dynamic state list (the spec ignores the states a part
/// does not own). A part is keyed by an XXH3-128 of exactly the state it reads; shader modules
/// count by program permutation (key stage hash) and handle, the generated auxiliary stages
/// (rect/quad list tessellation, clip-distance discard) by their SPIR-V. A create info carrying
/// a structure the hash does not know (a feature added to the monolithic path later) is not
/// linked: that pipeline compiles as before.
///
/// Set 0 holds the fragment stage's bindings, then the other stages' (one set, no independent
/// sets), so the stage parts must be built with an identically defined layout and their keys
/// include the whole layout. On the shader cache of a 22-minute Killzone session (4,153
/// pipelines, 522 layouts) that costs little: 2,352 pre-rasterization and 1,798 fragment parts
/// instead of 2,283 and 1,683 without the layout.
///
/// Libraries and linked pipelines never go through VK_KHR_pipeline_binary (binaries of libraries
/// are not portable the same way); they use the driver's VkPipelineCache, which the driver cache
/// file keeps between sessions. The optimized pipeline keeps using the binary cache, so the next
/// session restores it instead of linking.
///
/// GOW3_GPL=0 leaves the extension out of the device (vk_instance.cpp) and behaves as before.
/// GOW3_GPL_MAX_LIBRARIES (default 6000) caps the libraries kept (a full play needs ~4,200).
class PipelineLibraries {
public:
    enum Part : u32 { VertexInput, PreRaster, FragmentShader, FragmentOutput, NumParts };
    struct Key {
        u64 lo = 0, hi = 0;
        bool operator==(const Key&) const noexcept = default;
    };
    struct KeyHash {
        size_t operator()(const Key& key) const noexcept {
            return size_t(key.lo ^ (key.hi * 0x9E3779B97F4A7C15ull));
        }
    };

    explicit PipelineLibraries(const Instance& instance);
    ~PipelineLibraries();

    [[nodiscard]] bool Enabled() const noexcept {
        return enabled;
    }
    /// Counts shader-stage parts that became ready: a probe that missed is worth repeating only
    /// once it moved.
    [[nodiscard]] u64 Generation() const noexcept {
        return generation.load(std::memory_order_acquire);
    }
    [[nodiscard]] size_t Count() const;

    /// Library work on any thread, read by the report.
    struct Counters {
        std::array<std::atomic<u32>, NumParts> compiled{};
        std::array<std::atomic<u64>, NumParts> compile_ns{};
        std::array<std::atomic<u64>, NumParts> compile_worst_ns{};
        std::atomic<u32> reused{};      ///< a part found ready
        std::atomic<u32> failed{};      ///< a part or a link the driver refused
        std::atomic<u32> links{};
        std::atomic<u64> link_ns{}, link_worst_ns{};
        std::atomic<u64> probe_key_ns{};  ///< pipeline binary keys computed by probes
        std::atomic<u32> probe_keys{};
    };
    Counters counters;

private:
    friend class PipelineLinker;
    struct Entry {
        vk::UniquePipeline pipeline;
        bool done = false;
    };
    /// The ready library for `key`, or null. Creates it from `info` on this thread when nobody
    /// has (`may_create`), waits for another thread's creation (`may_wait`).
    vk::Pipeline Acquire(Part part, const Key& key, const vk::GraphicsPipelineCreateInfo& info,
                         vk::PipelineCache cache, bool may_create, bool may_wait);
    vk::ResultValue<vk::UniquePipeline> CreateLibrary(Part part,
                                                      const vk::GraphicsPipelineCreateInfo& info,
                                                      vk::PipelineCache cache) const;

    const Instance& instance;
    bool enabled = false;
    size_t max_libraries = 6000;
    mutable std::mutex mutex;
    std::condition_variable cv;
    std::array<std::unordered_map<Key, std::unique_ptr<Entry>, KeyHash>, NumParts> entries;
    size_t total = 0;
    std::atomic<u64> generation{0};
};

/// One pipeline's link request, handed to GraphicsPipeline's constructor: it calls Create with
/// the create info it would compile (and again for the squaring pass). A null pipeline means
/// "not linked" (see outcome); the constructor then returns early and the caller drops it.
class PipelineLinker {
public:
    enum class Mode : u8 {
        /// GPU thread: link only when both shader parts are ready; the two cheap parts may be
        /// created here. Never waits, never compiles a shader.
        Probe,
        /// Compiler worker: create every missing part, then link.
        Compile,
    };
    enum class Outcome : u8 {
        None,
        Linked,
        MissingLibraries, ///< Probe: a shader part is not ready
        BinaryCached,     ///< Probe: a pipeline binary is on disk, restoring it is cheaper
        Unsupported,      ///< state the part hash does not know
        Cancelled,        ///< Compile: the optimized pipeline was published first
        Failed,
    };

    PipelineLinker(PipelineLibraries& libraries, Mode mode, vk::PipelineCache cache,
                   const GraphicsPipelineKey& key, std::span<const vk::ShaderModule> modules,
                   const GraphicsPipeline::SerializationSupport& sdata);
    ~PipelineLinker();
    PipelineLinker(const PipelineLinker&) = delete;
    PipelineLinker& operator=(const PipelineLinker&) = delete;

    vk::ResultValue<vk::UniquePipeline> Create(const GraphicsPipeline& pipeline,
                                               const vk::GraphicsPipelineCreateInfo& info,
                                               bool square);

    Outcome outcome = Outcome::None;
    bool check_binary = false;              ///< Probe: BinaryCached when one is on disk
    const std::atomic<bool>* cancel = nullptr; ///< Compile: stop before the next shader part

private:
    bool HashPart(PipelineLibraries::Part part, const GraphicsPipeline& pipeline,
                  const vk::GraphicsPipelineCreateInfo& info, PipelineLibraries::Key& out);
    bool HashModule(PartHasher& h, const vk::PipelineShaderStageCreateInfo& stage);
    bool PipelineBinaryOnDisk(const vk::GraphicsPipelineCreateInfo& info) const;

    PipelineLibraries& libraries;
    Mode mode;
    vk::PipelineCache cache;
    std::array<u64, MaxShaderStages> stage_hashes{};
    std::array<vk::ShaderModule, MaxShaderStages> modules{};
    const GraphicsPipeline::SerializationSupport& sdata;
    /// Generated stages (CompileSPV in the constructor, never freed there): freed with the
    /// linker, the parts and links made from them no longer need them.
    std::vector<vk::ShaderModule> aux_modules;
};

/// gow3: PipelineCache's fast-link state and the "Shaders:" timing report. GPU thread unless
/// noted; the build objects are the compiler jobs' (vk_pipeline_cache.cpp).
struct FastLinkState {
    explicit FastLinkState(const Instance& instance) : libraries{instance} {}

    PipelineLibraries libraries;
    /// Fast-link jobs in flight (urgent queue), by pipeline key.
    tsl::robin_map<GraphicsPipelineKey, std::shared_ptr<AsyncGraphicsCompilation>> pending;
    /// Keys never fast-linked: a binary on disk (their optimized compile is queued urgent),
    /// unsupported state, or a failed link.
    tsl::robin_set<GraphicsPipelineKey> ineligible;
    tsl::robin_set<GraphicsPipelineKey> urgent;
    /// Probe found a shader part missing at this library generation.
    tsl::robin_map<GraphicsPipelineKey, u64> missing;
    /// Fast-linked pipelines replaced by the optimized one, or never used (the optimized came
    /// first): kept until shutdown. A replaced one may still be referenced by recorded or queued
    /// draws, and the parts made with a pipeline's layout outlive it.
    std::vector<std::unique_ptr<GraphicsPipeline>> kept;

    struct Timing {
        u32 count = 0;
        double total_ms = 0.0, worst_ms = 0.0;
        void Add(double ms) {
            ++count;
            total_ms += ms;
            worst_ms = ms > worst_ms ? ms : worst_ms;
        }
    };
    // Report counters, reset by each report.
    u32 linked_now = 0, linked_async = 0, replaced = 0, optimized_first = 0;
    u32 binary_first = 0, unsupported = 0, failed = 0;
    u32 pending_shader = 0, pending_pipeline = 0;
    Timing ready_shader, ready_fast, ready_optimized;
    std::array<u32, PipelineLibraries::NumParts> compiled_seen{};
    std::array<u64, PipelineLibraries::NumParts> compile_ns_seen{};
    u32 reused_seen = 0, links_seen = 0, probe_keys_seen = 0;
    u64 link_ns_seen = 0, probe_key_ns_seen = 0;
    std::chrono::steady_clock::time_point last_report{};

    /// Every 5 s with activity: "Shaders: fast link ..." and "Shaders: ready after ...".
    void Report();
};

} // namespace Vulkan
