#include "Resource/Texture2D.hpp"

#include <RHI/RHIContext.hpp>

#include <fstream>

namespace Ilum
{
struct Resource<ResourceType::Texture2D>::Impl
{
	std::unique_ptr<RHITexture> texture = nullptr;
};

Resource<ResourceType::Texture2D>::Resource(RHIContext *rhi_context, const std::string &name) :
    IResource(rhi_context, name, ResourceType::Texture2D)
{
}

Resource<ResourceType::Texture2D>::Resource(RHIContext *rhi_context, std::vector<uint8_t> &&data, const TextureDesc &desc) :
    IResource(desc.name)
{
	m_impl = std::make_unique<Impl>();

	m_impl->texture = rhi_context->CreateTexture(desc);
	m_thumbnail     = rhi_context->CreateTexture2D(128, 128, RHIFormat::R8G8B8A8_UNORM, RHITextureUsage::ShaderResource | RHITextureUsage::Transfer, false);

	BufferDesc buffer_desc = {};
	buffer_desc.size       = glm::max(data.size(), 4ull * 128ull * 128ull);
	buffer_desc.usage      = RHIBufferUsage::Transfer;
	buffer_desc.memory     = RHIMemoryUsage::CPU_TO_GPU;

	auto staging_buffer = rhi_context->CreateBuffer(buffer_desc);
	std::memcpy(staging_buffer->Map(), data.data(), buffer_desc.size);
	staging_buffer->Unmap();

	auto *cmd_buffer = rhi_context->CreateCommand(RHIQueueFamily::Graphics);
	cmd_buffer->Begin();
	cmd_buffer->ResourceStateTransition(
	    {TextureStateTransition{
	         m_impl->texture.get(),
	         RHIResourceState::Undefined,
	         RHIResourceState::TransferDest,
	         TextureRange{RHITextureDimension::Texture2D, 0, m_impl->texture.get()->GetDesc().mips, 0, 1}},
	     TextureStateTransition{
	         m_thumbnail.get(),
	         RHIResourceState::Undefined,
	         RHIResourceState::TransferDest,
	         TextureRange{RHITextureDimension::Texture2D, 0, 1, 0, 1}}},
	    {});
	cmd_buffer->CopyBufferToTexture(staging_buffer.get(), m_impl->texture.get(), 0, 0, 1);
	cmd_buffer->GenerateMipmaps(m_impl->texture.get(), RHIResourceState::Undefined, RHIFilter::Linear);
	cmd_buffer->BlitTexture(m_impl->texture.get(), TextureRange{RHITextureDimension::Texture2D, 0, 1, 0, 1}, RHIResourceState::TransferDest,
	                        m_thumbnail.get(), TextureRange{RHITextureDimension::Texture2D, 0, 1, 0, 1}, RHIResourceState::TransferDest);
	cmd_buffer->ResourceStateTransition(
	    {TextureStateTransition{
	         m_impl->texture.get(),
	         RHIResourceState::TransferDest,
	         RHIResourceState::ShaderResource,
	         TextureRange{RHITextureDimension::Texture2D, 0, m_impl->texture.get()->GetDesc().mips, 0, 1}},
	     TextureStateTransition{
	         m_thumbnail.get(),
	         RHIResourceState::TransferDest,
	         RHIResourceState::TransferSource,
	         TextureRange{RHITextureDimension::Texture2D, 0, 1, 0, 1}}},
	    {});
	cmd_buffer->CopyTextureToBuffer(m_thumbnail.get(), staging_buffer.get(), 0, 0, 1);
	cmd_buffer->End();

	rhi_context->Execute(cmd_buffer);

	std::vector<uint8_t> thumbnail_data(4 * 128 * 128);
	std::memcpy(thumbnail_data.data(), staging_buffer->Map(), thumbnail_data.size());
	staging_buffer->Unmap();

	SERIALIZE(fmt::format("Asset/Meta/{}.{}.asset", m_name, (uint32_t) ResourceType::Texture2D), thumbnail_data, desc, data);
}

Resource<ResourceType::Texture2D>::~Resource()
{
	m_impl.reset();
}

bool Resource<ResourceType::Texture2D>::Validate() const
{
	return m_impl != nullptr;
}

void Resource<ResourceType::Texture2D>::Load(RHIContext *rhi_context)
{
	std::vector<uint8_t> thumbnail_data, data;
	TextureDesc          desc;

	DESERIALIZE(fmt::format("Asset/Meta/{}.{}.asset", m_name, (uint32_t) ResourceType::Texture2D), thumbnail_data, desc, data);

	m_impl = std::make_unique<Impl>();

	m_impl->texture = rhi_context->CreateTexture(desc);

	BufferDesc buffer_desc = {};
	buffer_desc.size       = data.size();
	buffer_desc.usage      = RHIBufferUsage::Transfer;
	buffer_desc.memory     = RHIMemoryUsage::CPU_TO_GPU;

	auto staging_buffer = rhi_context->CreateBuffer(buffer_desc);
	std::memcpy(staging_buffer->Map(), data.data(), buffer_desc.size);
	staging_buffer->Unmap();

	auto *cmd_buffer = rhi_context->CreateCommand(RHIQueueFamily::Graphics);
	cmd_buffer->Begin();
	cmd_buffer->ResourceStateTransition(
	    {TextureStateTransition{
	        m_impl->texture.get(),
	        RHIResourceState::Undefined,
	        RHIResourceState::TransferDest,
	        TextureRange{RHITextureDimension::Texture2D, 0, m_impl->texture.get()->GetDesc().mips, 0, 1}}},
	    {});
	cmd_buffer->CopyBufferToTexture(staging_buffer.get(), m_impl->texture.get(), 0, 0, 1);
	cmd_buffer->GenerateMipmaps(m_impl->texture.get(), RHIResourceState::Undefined, RHIFilter::Linear);
	cmd_buffer->ResourceStateTransition(
	    {TextureStateTransition{
	        m_impl->texture.get(),
	        RHIResourceState::TransferDest,
	        RHIResourceState::ShaderResource,
	        TextureRange{RHITextureDimension::Texture2D, 0, m_impl->texture.get()->GetDesc().mips, 0, 1}}},
	    {});
	cmd_buffer->End();

	rhi_context->Execute(cmd_buffer);
}

RHITexture *Resource<ResourceType::Texture2D>::GetTexture() const
{
	return m_impl->texture.get();
}
}        // namespace Ilum