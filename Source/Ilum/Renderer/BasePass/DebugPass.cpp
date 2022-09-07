#include "DebugPass.hpp"

namespace Ilum
{
RenderPassDesc DebugPass::CreateDesc()
{
	RenderPassDesc desc;
	desc.name = "DebugPass";
	desc.Read("Debug", RenderResourceDesc::Type::Texture, RHIResourceState::TransferSource);
	return desc;
}

RenderGraph::RenderTask DebugPass::Create(const RenderPassDesc &desc, RenderGraphBuilder &builder, Renderer *renderer)
{
	std::shared_ptr<RHITexture> debug_texture = renderer->GetRHIContext()->CreateTexture2D(100, 100, RHIFormat::R8G8B8A8_UNORM, RHITextureUsage::ShaderResource | RHITextureUsage::Transfer, false);

	{
		auto cmd_buffer = renderer->GetRHIContext()->CreateCommand(RHIQueueFamily::Graphics);
		cmd_buffer->Begin();
		cmd_buffer->ResourceStateTransition({TextureStateTransition{
		                                        debug_texture.get(),
		                                        RHIResourceState::Undefined,
		                                        RHIResourceState::ShaderResource,
		                                        TextureRange{RHITextureDimension::Texture2D, 0, 1, 0, 1}}},
		                                    {});
		cmd_buffer->End();
		renderer->GetRHIContext()->GetQueue(RHIQueueFamily::Graphics)->Submit({cmd_buffer});
	}

	return [=](RenderGraph &render_graph, RHICommand *cmd_buffer, rttr::variant &config) {
		config = debug_texture.get();
		auto texture = render_graph.GetTexture(desc.resources.at("Debug").handle);
		cmd_buffer->ResourceStateTransition({TextureStateTransition{
		                                        debug_texture.get(),
		                                        RHIResourceState::ShaderResource,
		                                        RHIResourceState::TransferDest,
		                                        TextureRange{RHITextureDimension::Texture2D, 0, 1, 0, 1}}},
		                                    {});
		cmd_buffer->BlitTexture(
		    texture,
		    TextureRange{RHITextureDimension::Texture2D, 0, 1, 0, 1},
		    RHIResourceState::TransferSource,
		    debug_texture.get(),
		    TextureRange{RHITextureDimension::Texture2D, 0, 1, 0, 1},
		    RHIResourceState::TransferDest);
		cmd_buffer->ResourceStateTransition({TextureStateTransition{
		                                        debug_texture.get(),
		                                        RHIResourceState::TransferDest,
		                                        RHIResourceState::ShaderResource,
		                                        TextureRange{RHITextureDimension::Texture2D, 0, 1, 0, 1}}},
		                                    {});
	};
}
}        // namespace Ilum