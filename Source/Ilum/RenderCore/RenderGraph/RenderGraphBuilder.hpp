#pragma once

#include "RenderGraph.hpp"

namespace Ilum
{
class RenderGraph;

template <typename T>
void variadic_vector_emplace(std::vector<T> &)
{}

template <typename T, typename First, typename... Args>
void variadic_vector_emplace(std::vector<T> &v, First &&first, Args &&...args)
{
	v.emplace_back(std::forward<First>(first));
	variadic_vector_emplace(v, std::forward<Args>(args)...);
}

class RenderGraphBuilder
{
  public:
	RenderGraphBuilder(RHIContext *rhi_context);

	~RenderGraphBuilder() = default;

	RenderGraphBuilder &AddPass(RenderGraph &render_graph, const std::string &name, BindPoint bind_point, const rttr::variant &config, RenderGraph::RenderTask &&task, RenderGraph::BarrierTask &&barrier);

	bool Validate(RenderGraphDesc &desc);

	std::unique_ptr<RenderGraph> Compile();

	template <typename... Args>
	std::unique_ptr<RenderGraph> Compile(RenderGraphDesc &desc, Args &&...args);

  private:
	RHIContext *p_rhi_context = nullptr;
};
}        // namespace Ilum

#include "RenderGraphBuilder.inl"