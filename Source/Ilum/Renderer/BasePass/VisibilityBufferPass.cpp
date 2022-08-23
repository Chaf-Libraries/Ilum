#include "VisibilityBufferPass.hpp"

namespace Ilum::Pass
{
RenderPassDesc VisibilityBufferPass::CreateDesc(size_t &handle)
{
	RenderPassDesc desc = {};

	desc.name = "VisibilityBufferPass";
	desc.writes.emplace("VisibilityBuffer", std::make_pair(RenderPassDesc::ResourceType::Texture, RGHandle(handle++)));
	desc.writes.emplace("DepthBuffer", std::make_pair(RenderPassDesc::ResourceType::Texture, RGHandle(handle++)));

	desc.variant = rttr::type::get_by_name("VisibilityBufferPass::Config").create();

	return desc;
}

void VisibilityBufferPass::Create(const RenderPassDesc &desc, RenderGraphBuilder &builder, Renderer *renderer)
{
	builder.AddPass("Visibility Buffer Pass", [desc](RenderGraph &render_graph) {
		RGHandle visibility_buffer_handle = desc.writes.at("VisibilityBuffer").second;
		RGHandle depth_buffer_handle      = desc.writes.at("DepthBuffer").second;

		// auto visibility_buffer = render_graph.GetTexture("VisibilityBuffer");
		// auto depth_buffer      = render_graph.GetTexture("DepthBuffer");

		Config config = desc.variant.convert<Config>();
	});
}

}        // namespace Ilum::Pass