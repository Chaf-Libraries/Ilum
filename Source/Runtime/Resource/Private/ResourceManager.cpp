#include "ResourceManager.hpp"
#include "Importer/Texture/TextureImporter.hpp"

#include <Core/Path.hpp>
#include <RHI/RHIContext.hpp>

namespace Ilum
{
static const TextureDesc ThumbnailDesc = {"", 48, 48, 1, 1, 1, 1, RHIFormat::R8G8B8A8_UNORM, RHITextureUsage::Transfer | RHITextureUsage::ShaderResource};

struct IResourceManager
{
  public:
	virtual Resource *GetResource(size_t uuid) = 0;

	virtual void EraseResource(size_t uuid) = 0;

	virtual bool HasResource(size_t uuid) = 0;

	virtual void AddResource(std::unique_ptr<Resource> &&resource) = 0;
};

template <ResourceType _Ty>
struct TResourceManager : public IResourceManager
{
	std::vector<std::unique_ptr<TResource<_Ty>>> m_resource;
	std::map<size_t, size_t>                     m_resource_lookup;

	virtual Resource *GetResource(size_t uuid) override
	{
		if (m_resource_lookup.find(uuid) != m_resource_lookup.end())
		{
			return m_resource[m_resource_lookup.at(uuid)].get();
		}
		return nullptr;
	}

	virtual void EraseResource(size_t uuid) override
	{
		if (m_resource_lookup.find(uuid) != m_resource_lookup.end())
		{
			size_t last_uuid             = m_resource.back()->GetUUID();
			if (m_resource.size() > 1)
			{
				m_resource_lookup[last_uuid] = m_resource_lookup[uuid];
				std::swap(m_resource.back(), m_resource[uuid]);
			}
			m_resource.pop_back();
			m_resource_lookup.erase(uuid);
			Path::GetInstance().DeletePath("Asset/Meta/" + std::to_string(uuid) + ".meta");
		}
	}

	virtual bool HasResource(size_t uuid) override
	{
		return m_resource_lookup.find(uuid) != m_resource_lookup.end();
	}

	virtual void AddResource(std::unique_ptr<Resource> &&resource) override
	{
		m_resource_lookup.emplace(resource->GetUUID(), m_resource.size());
		TResource<_Ty> *ptr = static_cast<TResource<_Ty> *>(std::move(resource).release());
		m_resource.emplace_back(std::unique_ptr<TResource<_Ty>>(ptr));
	}
};

struct ResourceManager::Impl
{
	std::map<ResourceType, std::unique_ptr<IResourceManager>> m_managers;
};

ResourceManager::ResourceManager(RHIContext *rhi_context) :
    p_rhi_context(rhi_context), m_impl(std::make_unique<Impl>())
{
	m_impl->m_managers.emplace(ResourceType::Texture2D, std::make_unique<TResourceManager<ResourceType::Texture2D>>());
}

ResourceManager::~ResourceManager()
{
}

Resource *ResourceManager::GetResource(size_t uuid, ResourceType type)
{
	return m_impl->m_managers.at(type)->GetResource(uuid);
}

void ResourceManager::EraseResource(size_t uuid, ResourceType type)
{
	m_impl->m_managers.at(type)->EraseResource(uuid);
}

void ResourceManager::Import(const std::string &path, ResourceType type)
{
	switch (type)
	{
		case Ilum::ResourceType::Model:
			ImportModel(path);
			break;
		case Ilum::ResourceType::Texture2D:
			ImportTexture2D(path);
			break;
		default:
			break;
	}
}

void ResourceManager::ImportTexture2D(const std::string &path)
{
	size_t uuid = Hash(path);
	if (m_impl->m_managers.at(ResourceType::Texture2D)->HasResource(uuid))
	{
		LOG_INFO("{} is already cached", path);
		return;
	}

	TextureImportInfo info = TextureImporter::Import(path);

	std::string meta_info = fmt::format("Name: {}\nOriginal Path: {}\nWidth: {}\nHeight: {}\nMips: {}\nLayers: {}\nFormat: {}",
	                                    Path::GetInstance().GetFileName(path), path, info.desc.width, info.desc.height, info.desc.mips, info.desc.layers, rttr::type::get_by_name("Ilum::RHIFormat").get_enumeration().value_to_name(info.desc.format).to_string());

	SERIALIZE("Asset/Meta/" + std::to_string(uuid) + ".meta", ResourceType::Texture2D, uuid, meta_info, info);

	BufferDesc buffer_desc = {};
	buffer_desc.size       = info.data.size();
	buffer_desc.usage      = RHIBufferUsage::Transfer;
	buffer_desc.memory     = RHIMemoryUsage::CPU_TO_GPU;

	auto staging_buffer = p_rhi_context->CreateBuffer(buffer_desc);
	std::memcpy(staging_buffer->Map(), info.data.data(), buffer_desc.size);
	staging_buffer->Flush(0, buffer_desc.size);
	staging_buffer->Unmap();

	auto thumbnail_buffer = p_rhi_context->CreateBuffer(BufferDesc{"", RHIBufferUsage::Transfer, RHIMemoryUsage::GPU_TO_CPU, 48 * 48 * 4 * sizeof(uint8_t)});

	TextureDesc thumbnail_desc = ThumbnailDesc;
	thumbnail_desc.name        = Path::GetInstance().GetFileName(path, false) + " - thumbnail";

	std::unique_ptr<Resource> resource = std::make_unique<TResource<ResourceType::Texture2D>>(uuid, meta_info);

	auto *texture = static_cast<TResource<ResourceType::Texture2D> *>(resource.get());

	texture->SetTexture(p_rhi_context->CreateTexture(info.desc));
	texture->SetThumbnail(p_rhi_context->CreateTexture(thumbnail_desc));

	{
		auto *cmd_buffer = p_rhi_context->CreateCommand(RHIQueueFamily::Graphics);
		cmd_buffer->Begin();
		cmd_buffer->ResourceStateTransition({TextureStateTransition{
		                                         texture->GetTexture(),
		                                         RHIResourceState::Undefined,
		                                         RHIResourceState::TransferDest,
		                                         TextureRange{RHITextureDimension::Texture2D, 0, texture->GetTexture()->GetDesc().mips, 0, 1}},
		                                     TextureStateTransition{
		                                         texture->GetThumbnail(),
		                                         RHIResourceState::Undefined,
		                                         RHIResourceState::TransferDest,
		                                         TextureRange{RHITextureDimension::Texture2D, 0, 1, 0, 1}}},
		                                    {});
		cmd_buffer->CopyBufferToTexture(staging_buffer.get(), texture->GetTexture(), 0, 0, 1);
		cmd_buffer->GenerateMipmaps(texture->GetTexture(), RHIResourceState::Undefined, RHIFilter::Linear);
		cmd_buffer->ResourceStateTransition({TextureStateTransition{
		                                        texture->GetTexture(),
		                                        RHIResourceState::TransferDest,
		                                        RHIResourceState::TransferSource,
		                                        TextureRange{RHITextureDimension::Texture2D, 0, texture->GetTexture()->GetDesc().mips, 0, 1}}},
		                                    {});
		cmd_buffer->BlitTexture(
		    texture->GetTexture(),
		    TextureRange{RHITextureDimension::Texture2D, 0, 1, 0, 1},
		    RHIResourceState::TransferSource,
		    texture->GetThumbnail(),
		    TextureRange{RHITextureDimension::Texture2D, 0, 1, 0, 1},
		    RHIResourceState::TransferDest,
		    RHIFilter::Nearest);
		cmd_buffer->ResourceStateTransition({TextureStateTransition{
		                                         texture->GetTexture(),
		                                         RHIResourceState::TransferSource,
		                                         RHIResourceState::ShaderResource,
		                                         TextureRange{RHITextureDimension::Texture2D, 0, texture->GetTexture()->GetDesc().mips, 0, 1}},
		                                     TextureStateTransition{
		                                         texture->GetThumbnail(),
		                                         RHIResourceState::TransferDest,
		                                         RHIResourceState::TransferSource,
		                                         TextureRange{RHITextureDimension::Texture2D, 0, 1, 0, 1}}},
		                                    {});
		cmd_buffer->CopyTextureToBuffer(texture->GetThumbnail(), thumbnail_buffer.get(), 0, 0, 1);
		cmd_buffer->ResourceStateTransition({TextureStateTransition{
		                                        texture->GetThumbnail(),
		                                        RHIResourceState::TransferSource,
		                                        RHIResourceState::ShaderResource,
		                                        TextureRange{RHITextureDimension::Texture2D, 0, 1, 0, 1}}},
		                                    {});
		cmd_buffer->End();

		p_rhi_context->Execute(cmd_buffer);
	}

	info.thumbnail_data.resize(thumbnail_buffer->GetDesc().size);
	std::memcpy(info.thumbnail_data.data(), thumbnail_buffer->Map(), thumbnail_buffer->GetDesc().size);
	thumbnail_buffer->Unmap();

	m_impl->m_managers.at(ResourceType::Texture2D)->AddResource(std::move(resource));
}

void ResourceManager::ImportModel(const std::string &path)
{
}
}        // namespace Ilum