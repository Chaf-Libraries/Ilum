#include "Resource/Texture.hpp"

#include <RHI/RHIContext.hpp>

namespace Ilum
{
struct Resource<ResourceType::Texture>::Impl
{
	std::unique_ptr<RHITexture> texture = nullptr;
};

Resource<ResourceType::Texture>::Resource(RHIContext *rhi_context, std::vector<uint8_t> &&data, const TextureDesc &desc)
{
	m_impl = new Impl;

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
	cmd_buffer->ResourceStateTransition({TextureStateTransition{
	                                        m_impl->texture.get(),
	                                        RHIResourceState::Undefined,
	                                        RHIResourceState::TransferDest,
	                                        TextureRange{RHITextureDimension::Texture2D, 0, m_impl->texture.get()->GetDesc().mips, 0, 1}}},
	                                    {});
	cmd_buffer->CopyBufferToTexture(staging_buffer.get(), m_impl->texture.get(), 0, 0, 1);
	cmd_buffer->GenerateMipmaps(m_impl->texture.get(), RHIResourceState::Undefined, RHIFilter::Linear);
	cmd_buffer->ResourceStateTransition({TextureStateTransition{
	                                        m_impl->texture.get(),
	                                        RHIResourceState::TransferDest,
	                                        RHIResourceState::ShaderResource,
	                                        TextureRange{RHITextureDimension::Texture2D, 0, m_impl->texture.get()->GetDesc().mips, 0, 1}}},
	                                    {});
	cmd_buffer->End();

	rhi_context->Execute(cmd_buffer);
}

Resource<ResourceType::Texture>::~Resource()
{
	delete m_impl;
}

RHITexture *Resource<ResourceType::Texture>::GetTexture() const
{
	return m_impl->texture.get();
}
}        // namespace Ilum