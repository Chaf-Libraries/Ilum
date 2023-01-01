#include "Resource.hpp"
#include "Resource/Animation.hpp"
#include "Resource/Material.hpp"
#include "Resource/Mesh.hpp"
#include "Resource/Prefab.hpp"
#include "Resource/SkinnedMesh.hpp"
#include "Resource/Texture2D.hpp"

#include <RHI/RHIContext.hpp>

namespace Ilum
{
IResource::IResource(const std::string &name) :
    m_name(name)
{
}

IResource::IResource(RHIContext *rhi_context, const std::string &name, ResourceType type) :
    m_name(name)
{
	std::vector<uint8_t> thumbnail_data;
	DESERIALIZE(fmt::format("Asset/Meta/{}.{}.asset", m_name, (uint32_t) type), thumbnail_data);

	if (thumbnail_data.empty())
	{
		return;
	}

	m_thumbnail = rhi_context->CreateTexture2D(128, 128, RHIFormat::R8G8B8A8_UNORM, RHITextureUsage::ShaderResource | RHITextureUsage::Transfer, false);

	BufferDesc buffer_desc = {};
	buffer_desc.size       = thumbnail_data.size();
	buffer_desc.usage      = RHIBufferUsage::Transfer;
	buffer_desc.memory     = RHIMemoryUsage::CPU_TO_GPU;

	auto staging_buffer = rhi_context->CreateBuffer(buffer_desc);
	std::memcpy(staging_buffer->Map(), thumbnail_data.data(), buffer_desc.size);
	staging_buffer->Unmap();

	auto *cmd_buffer = rhi_context->CreateCommand(RHIQueueFamily::Graphics);
	cmd_buffer->Begin();
	cmd_buffer->ResourceStateTransition(
	    {TextureStateTransition{
	        m_thumbnail.get(),
	        RHIResourceState::Undefined,
	        RHIResourceState::TransferDest,
	        TextureRange{RHITextureDimension::Texture2D, 0, 1, 0, 1}}},
	    {});
	cmd_buffer->CopyBufferToTexture(staging_buffer.get(), m_thumbnail.get(), 0, 0, 1);
	cmd_buffer->GenerateMipmaps(m_thumbnail.get(), RHIResourceState::Undefined, RHIFilter::Linear);
	cmd_buffer->ResourceStateTransition(
	    {TextureStateTransition{
	        m_thumbnail.get(),
	        RHIResourceState::TransferDest,
	        RHIResourceState::ShaderResource,
	        TextureRange{RHITextureDimension::Texture2D, 0, 1, 0, 1}}},
	    {});
	cmd_buffer->End();

	rhi_context->Execute(cmd_buffer);
}

const std::string &IResource::GetName() const
{
	return m_name;
}

size_t IResource::GetUUID() const
{
	return Hash(m_name);
}

RHITexture *IResource::GetThumbnail() const
{
	return m_thumbnail ? m_thumbnail.get() : nullptr;
}

template class EXPORT_API Resource<ResourceType::Mesh>;
template class EXPORT_API Resource<ResourceType::SkinnedMesh>;
template class EXPORT_API Resource<ResourceType::Material>;
template class EXPORT_API Resource<ResourceType::Texture2D>;
template class EXPORT_API Resource<ResourceType::Prefab>;
template class EXPORT_API Resource<ResourceType::Animation>;
}        // namespace Ilum