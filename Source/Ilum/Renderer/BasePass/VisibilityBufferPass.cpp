#include "VisibilityBufferPass.hpp"

namespace Ilum::Pass
{
RenderPassDesc VisibilityBufferPass::CreateDesc()
{
	RenderPassDesc desc = {};

	desc.name = "VisibilityBufferPass";
	desc
	    .Write("VisibilityBuffer", RenderPassDesc::ResourceInfo::Type::Texture, RHIResourceState::RenderTarget)
	    .Write("DepthBuffer", RenderPassDesc::ResourceInfo::Type::Texture, RHIResourceState::RenderTarget);

	desc.variant = rttr::type::get_by_name("VisibilityBufferPass::Config").create();

	return desc;
}

RenderGraph::RenderTask VisibilityBufferPass::Create(const RenderPassDesc &desc, RenderGraphBuilder &builder, Renderer *renderer)
{
	return [=](RenderGraph &render_graph, RHICommand * cmd_buffer) {
		RGHandle visibility_buffer_handle = desc.writes.at("VisibilityBuffer").handle;
		RGHandle depth_buffer_handle      = desc.writes.at("DepthBuffer").handle;

		// auto visibility_buffer = render_graph.GetTexture("VisibilityBuffer");
		// auto depth_buffer      = render_graph.GetTexture("DepthBuffer");

		Config config = desc.variant.convert<Config>();
	};
}

}        // namespace Ilum::Pass