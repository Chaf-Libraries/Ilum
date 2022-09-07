#include "PresentPass.hpp"

namespace Ilum
{
RenderPassDesc PresentPass::CreateDesc()
{
	RenderPassDesc desc;
	desc.name = "PresentPass";
	desc
	    .Read("Present", RenderResourceDesc::Type::Texture, RHIResourceState::ShaderResource);

	return desc;
}

RenderGraph::RenderTask PresentPass::Create(const RenderPassDesc &desc, RenderGraphBuilder &builder, Renderer *renderer)
{
	return [=](RenderGraph &render_graph, RHICommand *cmd_buffer, rttr::variant &config) {
		auto texture = render_graph.GetTexture(desc.resources.at("Present").handle);
		renderer->SetPresentTexture(texture);
	};
}
}        // namespace Ilum