#include "VisibilityBufferPass.hpp"

namespace Ilum
{
RenderPassDesc VisibilityBufferPass::CreateDesc()
{
	RenderPassDesc desc = {};

	desc.name = "VisibilityBufferPass";
	desc
	    .Write("VisibilityBuffer", RenderResourceDesc::Type::Texture, RHIResourceState::RenderTarget)
	    .Write("DepthBuffer", RenderResourceDesc::Type::Texture, RHIResourceState::RenderTarget);

	 desc.config = Config();

	return desc;
}

RenderGraph::RenderTask VisibilityBufferPass::Create(const RenderPassDesc &desc, RenderGraphBuilder &builder, Renderer *renderer)
{
	return [=](RenderGraph &render_graph, RHICommand *cmd_buffer, rttr::variant& config) {
		RGHandle visibility_buffer_handle = desc.resources.at("VisibilityBuffer").handle;
		RGHandle depth_buffer_handle      = desc.resources.at("DepthBuffer").handle;

		auto *visibility_buffer = render_graph.GetTexture(visibility_buffer_handle);
		auto *depth_buffer = render_graph.GetTexture(depth_buffer_handle);

		// auto visibility_buffer = render_graph.GetTexture("VisibilityBuffer");
		// auto depth_buffer      = render_graph.GetTexture("DepthBuffer");

		// Config config = desc.config.convert<VisibilityBufferPass_Config>();
	};
}

}        // namespace Ilum