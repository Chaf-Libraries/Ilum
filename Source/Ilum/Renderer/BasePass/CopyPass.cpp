#include "CopyPass.hpp"

namespace Ilum
{
RenderPassDesc CopyPass::CreateDesc()
{
	RenderPassDesc desc = {};

	desc.name = "CopyPass";
	desc
	    .Read("Source", RenderResourceDesc::Type::Texture, RHIResourceState::ShaderResource)
	    .Write("Target", RenderResourceDesc::Type::Texture, RHIResourceState::RenderTarget);

	return desc;
}

RenderGraph::RenderTask CopyPass::Create(const RenderPassDesc &desc, RenderGraphBuilder &builder, Renderer *renderer)
{
	return [=](RenderGraph &render_graph, RHICommand *cmd_buffer) {};
}
}        // namespace Ilum