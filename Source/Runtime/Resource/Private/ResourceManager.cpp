#include "ResourceManager.hpp"
#include "Importer/Texture/TextureImporter.hpp"

#include <Core/Path.hpp>
#include <RHI/RHIContext.hpp>

#include <filesystem>

namespace Ilum
{
inline void LoadTextureFromBuffer(RHIContext *rhi_context, std::unique_ptr<RHITexture> &texture, const TextureDesc &desc, const std::vector<uint8_t> &data)
{
	texture = rhi_context->CreateTexture(desc);

	auto staging_buffer = rhi_context->CreateBuffer(BufferDesc{"", RHIBufferUsage::Transfer, RHIMemoryUsage::GPU_TO_CPU, data.size()});
	std::memcpy(staging_buffer->Map(), data.data(), data.size());
	staging_buffer->Flush(0, data.size());
	staging_buffer->Unmap();

	auto *cmd_buffer = rhi_context->CreateCommand(RHIQueueFamily::Graphics);
	cmd_buffer->Begin();
	cmd_buffer->ResourceStateTransition({TextureStateTransition{
	                                        texture.get(),
	                                        RHIResourceState::Undefined,
	                                        RHIResourceState::TransferDest,
	                                        TextureRange{RHITextureDimension::Texture2D, 0, 1, 0, 1}}},
	                                    {});
	cmd_buffer->CopyBufferToTexture(staging_buffer.get(), texture.get(), 0, 0, 1);
	cmd_buffer->ResourceStateTransition({TextureStateTransition{
	                                        texture.get(),
	                                        RHIResourceState::TransferDest,
	                                        RHIResourceState::ShaderResource,
	                                        TextureRange{RHITextureDimension::Texture2D, 0, 1, 0, 1}}},
	                                    {});
	cmd_buffer->End();

	rhi_context->Execute(cmd_buffer);
}

struct IResourceManager
{
  public:
	IResourceManager(RHIContext *rhi_context) :
	    p_rhi_context(rhi_context)
	{
	}

	virtual ~IResourceManager() = default;

	virtual Resource *GetResource(size_t uuid) = 0;

	virtual const std::string &GetResourceMeta(size_t uuid) = 0;

	virtual void EraseResource(size_t uuid) = 0;

	virtual bool HasResource(size_t uuid) = 0;

	virtual Resource *CreateResource(size_t uuid) = 0;

	virtual void CacheResource(size_t uuid, const std::string &meta) = 0;

	virtual size_t ResourceIndex(size_t uuid) = 0;

	virtual bool IsResourceValid(size_t uuid) = 0;

	const std::vector<size_t> &GetUUIDs() const
	{
		return m_uuids;
	}

	void Tick()
	{
		m_update = false;
	}

	bool IsUpdate()
	{
		return m_update;
	}

  protected:
	RHIContext *p_rhi_context = nullptr;

	std::vector<size_t> m_uuids;

	bool m_update = false;
};

template <ResourceType _Ty>
struct TResourceManager : public IResourceManager
{
	std::vector<std::unique_ptr<TResource<_Ty>>> m_resource;

	std::map<size_t, size_t> m_resource_lookup;

	std::vector<TResource<_Ty>*> m_valid_resource;

	std::vector<std::unique_ptr<TResource<_Ty>>> m_deprecate_resource;

	TResourceManager(RHIContext *rhi_context) :
	    IResourceManager(rhi_context)
	{
	}

	virtual ~TResourceManager() = default;

	virtual Resource *GetResource(size_t uuid) override
	{
		if (m_resource_lookup.find(uuid) != m_resource_lookup.end())
		{
			auto *resource = m_resource[m_resource_lookup.at(uuid)].get();
			if (!resource->IsValid())
			{
				resource->Load(p_rhi_context, m_valid_resource.size());
				m_valid_resource.push_back(resource);
			}
			return resource;
		}
		return nullptr;
	}

	virtual const std::string &GetResourceMeta(size_t uuid)
	{
		return m_resource.at(m_resource_lookup.at(uuid))->GetMeta();
	}

	virtual void EraseResource(size_t uuid) override
	{
		if (m_resource_lookup.find(uuid) != m_resource_lookup.end())
		{
			size_t last_uuid = m_resource.back()->GetUUID();
			auto  *resource  = m_resource[m_resource_lookup.at(uuid)].get();
			if (m_resource.size() > 1)
			{
				m_resource_lookup[last_uuid] = m_resource_lookup[uuid];
				std::swap(m_resource.back(), m_resource[m_resource_lookup[uuid]]);
			}
			if (std::find(m_valid_resource.begin(), m_valid_resource.end(), resource) != m_valid_resource.end())
			{
				m_valid_resource.erase(std::remove(m_valid_resource.begin(), m_valid_resource.end(), resource));
			}
			m_deprecate_resource.emplace_back(std::move(m_resource.back()));
			m_resource.pop_back();
			m_resource_lookup.erase(uuid);
			m_uuids.erase(std::remove(m_uuids.begin(), m_uuids.end(), uuid));
			Path::GetInstance().DeletePath("Asset/Meta/" + std::to_string(uuid) + ".asset");
		}
	}

	virtual bool HasResource(size_t uuid) override
	{
		return m_resource_lookup.find(uuid) != m_resource_lookup.end();
	}

	virtual Resource *CreateResource(size_t uuid) override
	{
		m_resource_lookup.emplace(uuid, m_resource.size());
		m_resource.emplace_back(std::make_unique<TResource<_Ty>>(uuid));
		m_uuids.push_back(uuid);
		return m_resource.back().get();
	}

	virtual void CacheResource(size_t uuid, const std::string &meta) override
	{
		m_resource_lookup.emplace(uuid, m_resource.size());
		m_resource.emplace_back(std::make_unique<TResource<_Ty>>(uuid, meta, p_rhi_context));
		m_uuids.push_back(uuid);
	}

	virtual size_t ResourceIndex(size_t uuid) override
	{
		return m_resource_lookup.at(uuid);
	}

	virtual bool IsResourceValid(size_t uuid) override
	{
		return m_resource_lookup.find(uuid) != m_resource_lookup.end() && m_resource.at(m_resource_lookup.at(uuid))->IsValid();
	}
};

struct ResourceManager::Impl
{
	std::map<ResourceType, std::unique_ptr<IResourceManager>>     m_managers;
	std::unordered_map<ResourceType, std::unique_ptr<RHITexture>> m_thumbnails;
};

ResourceManager::ResourceManager(RHIContext *rhi_context) :
    p_rhi_context(rhi_context), m_impl(std::make_unique<Impl>())
{
	m_impl->m_managers.emplace(ResourceType::Texture, std::make_unique<TResourceManager<ResourceType::Texture>>(p_rhi_context));
	m_impl->m_managers.emplace(ResourceType::Model, std::make_unique<TResourceManager<ResourceType::Model>>(p_rhi_context));
	m_impl->m_managers.emplace(ResourceType::Scene, std::make_unique<TResourceManager<ResourceType::Scene>>(p_rhi_context));
	m_impl->m_managers.emplace(ResourceType::RenderGraph, std::make_unique<TResourceManager<ResourceType::RenderGraph>>(p_rhi_context));

	{
		TextureImportInfo info;

		info           = TextureImporter::Import("Asset/Icon/scene.png");
		info.desc.mips = 1;
		LoadTextureFromBuffer(p_rhi_context, m_impl->m_thumbnails[ResourceType::Scene], info.desc, info.data);

		info           = TextureImporter::Import("Asset/Icon/render_graph.png");
		info.desc.mips = 1;
		LoadTextureFromBuffer(p_rhi_context, m_impl->m_thumbnails[ResourceType::RenderGraph], info.desc, info.data);

		info           = TextureImporter::Import("Asset/Icon/model.png");
		info.desc.mips = 1;
		LoadTextureFromBuffer(p_rhi_context, m_impl->m_thumbnails[ResourceType::Model], info.desc, info.data);

		info           = TextureImporter::Import("Asset/Icon/texture.png");
		info.desc.mips = 1;
		LoadTextureFromBuffer(p_rhi_context, m_impl->m_thumbnails[ResourceType::Texture], info.desc, info.data);
	}

	ScanLocalMeta();
}

ResourceManager::~ResourceManager()
{
	m_impl.reset();
}

void ResourceManager::Tick()
{
	for (auto &[type, manager] : m_impl->m_managers)
	{
		manager->Tick();
	}
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
	if (!Path::GetInstance().IsExist(path))
	{
		LOG_ERROR("Resource {} is not found", path);
		return;
	}

	size_t uuid = Hash(path);

	if (m_impl->m_managers.at(type)->HasResource(uuid))
	{
		LOG_INFO("Resource {} is already cached", path);
	}
	else
	{
		Resource *resource = m_impl->m_managers.at(type)->CreateResource(uuid);
		resource->Import(p_rhi_context, path);
		ScanLocalMeta();
	}
}

RHITexture *ResourceManager::GetThumbnail(ResourceType type)
{
	return m_impl->m_thumbnails.at(type).get();
}

size_t ResourceManager::GetResourceIndex(size_t uuid, ResourceType type)
{
	return m_impl->m_managers.at(type)->ResourceIndex(uuid);
}

const std::string &ResourceManager::GetResourceMeta(size_t uuid, ResourceType type)
{
	return m_impl->m_managers.at(type)->GetResourceMeta(uuid);
}

bool ResourceManager::IsUpdate(ResourceType type)
{
	return m_impl->m_managers.at(type)->IsUpdate();
}

bool ResourceManager::IsValid(size_t uuid, ResourceType type)
{
	return m_impl->m_managers.at(type)->IsResourceValid(uuid);
}

const std::vector<size_t> &ResourceManager::GetResourceUUID(ResourceType type) const
{
	return m_impl->m_managers.at(type)->GetUUIDs();
}

void ResourceManager::ScanLocalMeta()
{
	const std::string meta_path = "Asset/Meta";
	for (const auto &path : std::filesystem::directory_iterator(meta_path))
	{
		if (Path::GetInstance().GetFileExtension(path.path().string()) != ".asset")
		{
			continue;
		}
		ResourceType type = ResourceType::None;
		std::string  meta = "";
		size_t       uuid = 0;
		DESERIALIZE(path.path(), type, uuid, meta);
		if (!m_impl->m_managers.at(type)->HasResource(uuid))
		{
			m_impl->m_managers.at(type)->CacheResource(uuid, meta);
		}
	}
}
}        // namespace Ilum