#pragma once

#include "Precompile.hpp"
#include "RenderPass.hpp"

#include <RHI/RHIContext.hpp>

namespace Ilum
{
class RenderGraphBlackboard;

class  RenderGraphDesc
{
  public:
	RenderGraphDesc() = default;

	~RenderGraphDesc() = default;

	RenderGraphDesc &SetName(const std::string &name);

	RenderGraphDesc &AddPass(size_t handle, RenderPassDesc &&desc);

	void ErasePass(size_t handle);

	void EraseLink(size_t source, size_t target);

	RenderGraphDesc &Link(size_t source, size_t target);

	bool HasLink(size_t target) const;

	size_t LinkFrom(size_t target) const;

	std::set<size_t> LinkTo(size_t source) const;

	bool HasPass(size_t handle) const;

	RenderPassDesc &GetPass(size_t handle);

	const std::string &GetName() const;

	std::map<size_t, RenderPassDesc> &GetPasses();

	const std::map<size_t, size_t> &GetEdges() const;

	void Clear();

	template <typename Archive>
	void serialize(Archive &archive)
	{
		archive(m_name, m_passes, m_edges, m_pass_lookup);
	}

  private:
	std::string m_name;

	std::map<size_t, RenderPassDesc> m_passes;

	std::map<size_t, size_t> m_edges;        // Target - Source

	std::map<size_t, size_t> m_pass_lookup;        // Pin ID - Node ID
};

class  RenderGraph
{
	friend class RenderGraphBuilder;

  public:
	using RenderTask  = std::function<void(RenderGraph &, RHICommand *, Variant &, RenderGraphBlackboard &)>;
	using BarrierTask = std::function<void(RenderGraph &, RHICommand *)>;
	using InitializeBarrierTask = std::function<void(RenderGraph &, RHICommand *, RHICommand *)>;

	struct RenderPassInfo
	{
		std::string name;
		std::string category;

		BindPoint bind_point;

		Variant config;

		RenderTask  execute;
		BarrierTask barrier;

		std::unique_ptr<RHIProfiler> profiler = nullptr;
	};

  public:
	RenderGraph(RHIContext *rhi_context);

	~RenderGraph();

	// Return the old one
	std::unique_ptr<RHITexture> SetTexture(size_t handle, std::unique_ptr<RHITexture> &&texture);

	RHITexture *GetTexture(size_t handle);

	RHIBuffer *GetBuffer(size_t handle);

	RHITexture *GetCUDATexture(size_t handle);

	void Execute(RenderGraphBlackboard &black_board);

	const std::vector<RenderPassInfo> &GetRenderPasses() const;

  private:
	struct TextureCreateInfo
	{
		TextureDesc      desc;
		std::set<size_t> handles;
	};

	struct BufferCreateInfo
	{
		BufferDesc       desc;
		std::set<size_t> handles;
	};

	RenderGraph &AddPass(
	    const std::string &name,
	    const std::string &category,
	    BindPoint          bind_point,
	    const Variant     &config,
	    RenderTask       &&execute,
	    BarrierTask      &&barrier);

	RenderGraph &AddInitializeBarrier(InitializeBarrierTask &&barrier);

	// Without memory alias
	RenderGraph &RegisterTexture(const TextureCreateInfo &create_infos);

	// With memory alias
	RenderGraph &RegisterTexture(const std::vector<TextureCreateInfo> &create_info);

	RenderGraph &RegisterBuffer(const BufferCreateInfo &create_info);

	RHISemaphore *MapToCUDASemaphore(RHISemaphore *semaphore);

  private:
	struct Impl;
	Impl *m_impl = nullptr;
};
}        // namespace Ilum