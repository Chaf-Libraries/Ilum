#include "RawCUDATexture.cuh"
#include "RawCUDATexture.hpp"

#include <RHI/../../Private/Backend/CUDA/Command.hpp>
#include <RHI/../../Private/Backend/CUDA/Texture.hpp>
#include <RHI/RHIDevice.hpp>
#include <Renderer/Renderer.hpp>
#include <Resource/ResourceManager.hpp>

namespace Ilum
{
RenderPassDesc RawCUDATexture::CreateDesc()
{
	RenderPassDesc desc;

	return desc.SetName<RawCUDATexture>()
	    .SetBindPoint(BindPoint::CUDA)
	    .Write("Result", RenderResourceDesc::Type::Texture, RHIResourceState::UnorderedAccess);
}

RenderGraph::RenderTask RawCUDATexture::Create(const RenderPassDesc &desc, RenderGraphBuilder &builder, Renderer *renderer)
{
	size_t tex_id = renderer->GetResourceManager()->Import<ResourceType::Texture>("Asset/Texture/Default/default.png");

	auto texture = renderer->GetResourceManager()->GetResource<ResourceType::Texture>(tex_id)->GetTexture();

	std::shared_ptr<RHITexture> gfx_texture = std::move(renderer->GetRHIContext()->CreateTexture2D(
	    texture->GetDesc().width,
	    texture->GetDesc().height,
	    RHIFormat::R32G32B32A32_FLOAT, RHITextureUsage::Transfer, false, 1));
	{
		auto *cmd_buffer = renderer->GetRHIContext()->CreateCommand(RHIQueueFamily::Graphics);
		cmd_buffer->Begin();
		cmd_buffer->BlitTexture(texture, TextureRange{RHITextureDimension::Texture2D, 0, 1, 0, 1}, RHIResourceState::ShaderResource, gfx_texture.get(), TextureRange{RHITextureDimension::Texture2D, 0, 1, 0, 1}, RHIResourceState::Undefined, RHIFilter::Nearest);
		cmd_buffer->End();
		renderer->GetRHIContext()->Execute(cmd_buffer);
	}
	std::shared_ptr<RHITexture> cuda_input_texture = renderer->GetRHIContext()->MapToCUDATexture(gfx_texture.get());

	return [=](RenderGraph &render_graph, RHICommand *cmd_buffer, rttr::variant &config) {
		auto *result = render_graph.GetCUDATexture(desc.resources.at("Result").handle);

		auto cuda_texture = static_cast<CUDA::Texture *>(result);

		static_cast<CUDA::Command *>(cmd_buffer)->Execute([=]() {
			ExecuteRawCUDATextureKernal(cuda_texture->GetSurfaceHostHandle()[0], *static_cast<CUDA::Texture *>(cuda_input_texture.get())->GetTextureHandle(), result->GetDesc().width, result->GetDesc().height, 1, 8, 8, 1);
		});
	};
}
}        // namespace Ilum