#include "Resource/TextureResource.hpp"
#include "Importer/Texture/TextureImporter.hpp"

#include <Core/Path.hpp>
#include <RHI/RHIContext.hpp>

namespace Ilum
{
TResource<ResourceType::Texture>::TResource(size_t uuid) :
    Resource(uuid)
{
}

TResource<ResourceType::Texture>::TResource(size_t uuid, const std::string &meta, RHIContext *rhi_context) :
    Resource(uuid, meta, rhi_context)
{
	ResourceType         type = ResourceType::None;
	TextureDesc          desc;
	std::vector<uint8_t> thumbnail_data;
	DESERIALIZE("Asset/Meta/" + std::to_string(m_uuid) + ".asset", type, m_uuid, m_meta, desc);
}

void TResource<ResourceType::Texture>::Load(RHIContext *rhi_context, size_t index)
{
	ResourceType         type = ResourceType::None;
	TextureDesc          desc;
	std::vector<uint8_t> data;
	DESERIALIZE("Asset/Meta/" + std::to_string(m_uuid) + ".asset", type, m_uuid, m_meta, desc, data);

	m_texture = rhi_context->CreateTexture(desc);

	{
		BufferDesc buffer_desc = {};
		buffer_desc.size       = data.size();
		buffer_desc.usage      = RHIBufferUsage::Transfer;
		buffer_desc.memory     = RHIMemoryUsage::CPU_TO_GPU;

		auto staging_buffer = rhi_context->CreateBuffer(buffer_desc);
		std::memcpy(staging_buffer->Map(), data.data(), buffer_desc.size);
		staging_buffer->Flush(0, buffer_desc.size);
		staging_buffer->Unmap();

		auto *cmd_buffer = rhi_context->CreateCommand(RHIQueueFamily::Graphics);
		cmd_buffer->Begin();
		cmd_buffer->ResourceStateTransition({TextureStateTransition{
		                                        m_texture.get(),
		                                        RHIResourceState::Undefined,
		                                        RHIResourceState::TransferDest,
		                                        TextureRange{RHITextureDimension::Texture2D, 0, m_texture.get()->GetDesc().mips, 0, 1}}},
		                                    {});
		cmd_buffer->CopyBufferToTexture(staging_buffer.get(), m_texture.get(), 0, 0, 1);
		cmd_buffer->GenerateMipmaps(m_texture.get(), RHIResourceState::Undefined, RHIFilter::Linear);
		cmd_buffer->ResourceStateTransition({TextureStateTransition{
		                                        m_texture.get(),
		                                        RHIResourceState::TransferDest,
		                                        RHIResourceState::ShaderResource,
		                                        TextureRange{RHITextureDimension::Texture2D, 0, m_texture.get()->GetDesc().mips, 0, 1}}},
		                                    {});
		cmd_buffer->End();

		rhi_context->Execute(cmd_buffer);
	}

	m_valid = true;
	m_index = index;
}

void TResource<ResourceType::Texture>::Import(RHIContext *rhi_context, const std::string &path)
{
	size_t uuid = Hash(path);

	TextureImportInfo info = TextureImporter::Import(path);

	m_meta = fmt::format("Name: {}\nOriginal Path: {}\nWidth: {}\nHeight: {}\nMips: {}\nLayers: {}\nFormat: {}",
	                     Path::GetInstance().GetFileName(path), path, info.desc.width, info.desc.height, info.desc.mips, info.desc.layers, rttr::type::get_by_name("RHIFormat").get_enumeration().value_to_name(info.desc.format).to_string());

	SERIALIZE("Asset/Meta/" + std::to_string(uuid) + ".asset", ResourceType::Texture, uuid, m_meta, info.desc, info.data);
}

RHITexture *TResource<ResourceType::Texture>::GetTexture() const
{
	return m_texture.get();
}
}        // namespace Ilum