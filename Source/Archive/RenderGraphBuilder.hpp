#pragma once

#include "Precompile.hpp"

#include "RenderGraph.hpp"

namespace Ilum
{
class RenderGraph;

class EXPORT_API RenderGraphBuilder
{
  public:
	RenderGraphBuilder(RHIContext *rhi_context);

	~RenderGraphBuilder() = default;

	RenderGraphBuilder &AddPass(RenderGraph &render_graph, const std::string &name, BindPoint bind_point, const Variant &config, RenderGraph::RenderTask &&task, RenderGraph::BarrierTask &&barrier);

	bool Validate(RenderGraphDesc &desc);

	std::unique_ptr<RenderGraph> Compile();

	template <typename... Args>
	std::unique_ptr<RenderGraph> Compile(RenderGraphDesc &desc, Args &&...args);

  private:
	RHIContext *p_rhi_context = nullptr;
};
}        // namespace Ilum

#include "Impl/RenderGraphBuilder.inl"