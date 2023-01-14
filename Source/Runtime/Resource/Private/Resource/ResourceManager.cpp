#include "ResourceManager.hpp"
#include "Importer.hpp"
#include "Resource.hpp"
#include "Resource/Animation.hpp"
#include "Resource/Material.hpp"
#include "Resource/Mesh.hpp"
#include "Resource/Prefab.hpp"
#include "Resource/RenderPipeline.hpp"
#include "Resource/Scene.hpp"
#include "Resource/SkinnedMesh.hpp"
#include "Resource/Texture2D.hpp"

#include <RHI/RHIContext.hpp>

namespace Ilum
{
struct IResourceManager
{
	IResourceManager(RHIContext *rhi_context) :
	    rhi_context(rhi_context)
	{
	}

	virtual ~IResourceManager() = default;

	virtual void Tick() = 0;

	virtual IResource *Get(size_t uuid) = 0;

	virtual const std::string GetName(size_t uuid) = 0;

	virtual RHITexture *GetThumbnail(size_t uuid) = 0;

	virtual bool Valid(size_t uuid) = 0;

	virtual bool Has(size_t uuid) = 0;

	virtual size_t Index(size_t uuid) = 0;

	virtual void Erase(size_t uuid) = 0;

	virtual void Import(ResourceManager *manager, const std::string &path) = 0;

	virtual void Add(std::unique_ptr<IResource> &&resource, size_t uuid) = 0;

	virtual const std::vector<std::string> GetResources(bool only_valid) const = 0;

	virtual bool Update() const = 0;

	virtual void SetDirty() = 0;

	RHIContext *rhi_context = nullptr;
};

template <ResourceType _Ty>
struct TResourceManager : public IResourceManager
{
	TResourceManager(RHIContext *rhi_context) :
	    IResourceManager(rhi_context)
	{
	}

	virtual ~TResourceManager() = default;

	virtual void Tick() override
	{
		deprecates.clear();
		update = false;
	}

	virtual Resource<_Ty> *Get(size_t uuid) override
	{
		if (valid_lookup.find(uuid) != valid_lookup.end())
		{
			auto *resource = valid_resources.at(valid_lookup.at(uuid));
			return resource;
		}
		else if (resource_lookup.find(uuid) != resource_lookup.end())
		{
			auto &resource = resources[resource_lookup[uuid]];
			resource->Load(rhi_context);
			valid_lookup.emplace(uuid, valid_resources.size());
			valid_resources.push_back(resource.get());
			update = true;
			return resource.get();
		}
		return nullptr;
	}

	virtual const std::string GetName(size_t uuid) override
	{
		return Has(uuid) ? resources[resource_lookup[uuid]]->GetName() : "";
	}

	virtual RHITexture *GetThumbnail(size_t uuid) override
	{
		return Has(uuid) ? resources.at(resource_lookup.at(uuid))->GetThumbnail() : nullptr;
	}

	virtual bool Valid(size_t uuid) override
	{
		return valid_lookup.find(uuid) != valid_lookup.end();
	}

	virtual bool Has(size_t uuid) override
	{
		return resource_lookup.find(uuid) != resource_lookup.end();
	}

	virtual size_t Index(size_t uuid) override
	{
		return (Valid(uuid) || Get(uuid)) ? valid_lookup.at(uuid) : ~0U;
	}

	virtual void Erase(size_t uuid) override
	{
		if (valid_lookup.find(uuid) != valid_lookup.end())
		{
			size_t last_uuid = ~0U;
			if (valid_resources.size() > 1)
			{
				for (auto &[uuid_, index] : valid_lookup)
				{
					if (index + 1 == valid_lookup.size())
					{
						last_uuid = uuid_;
						break;
					}
				}

				valid_lookup[last_uuid] = valid_lookup[uuid];
				std::swap(valid_resources.back(), valid_resources[valid_lookup[uuid]]);
			}

			valid_resources.pop_back();
			valid_lookup.erase(uuid);
			update = true;
		}

		if (resource_lookup.find(uuid) != resource_lookup.end())
		{
			size_t last_uuid = ~0U;
			if (resources.size() > 1)
			{
				for (auto &[uuid_, index] : resource_lookup)
				{
					if (index + 1 == resource_lookup.size())
					{
						last_uuid = uuid_;
						break;
					}
				}

				resource_lookup[last_uuid] = resource_lookup[uuid];
				std::swap(resources.back(), resources[resource_lookup[uuid]]);
			}

			deprecates.emplace_back(std::move(resources.back()));
			resources.pop_back();
			resource_lookup.erase(uuid);
			update = true;
		}
	}

	virtual void Import(ResourceManager *manager, const std::string &path) override
	{
		Importer<_Ty>::Import(manager, path, rhi_context);
	}

	virtual void Add(std::unique_ptr<IResource> &&resource, size_t uuid) override
	{
		if (resource_lookup.find(uuid) == resource_lookup.end())
		{
			resource_lookup.emplace(uuid, resources.size());
			resources.emplace_back(std::unique_ptr<Resource<_Ty>>(dynamic_cast<Resource<_Ty> *>(resource.release())));

			if (resources.back()->Validate())
			{
				valid_lookup.emplace(uuid, valid_resources.size());
				valid_resources.push_back(resources.back().get());
			}

			update = true;
		}
	}

	virtual const std::vector<std::string> GetResources(bool only_valid) const override
	{
		std::vector<std::string> handles;
		if (only_valid)
		{
			handles.resize(valid_resources.size());
			std::transform(valid_resources.begin(), valid_resources.end(), handles.begin(), [](const auto &resource) {
				return resource->GetName();
			});
		}
		else
		{
			handles.resize(resources.size());
			std::transform(resources.begin(), resources.end(), handles.begin(), [](const auto &resource) {
				return resource->GetName();
			});
		}

		return handles;
	}

	virtual bool Update() const override
	{
		return update;
	}

	virtual void SetDirty() override
	{
		update = true;
	}

	std::vector<std::unique_ptr<Resource<_Ty>>> resources;
	std::unordered_map<size_t, size_t>          resource_lookup;

	std::vector<Resource<_Ty> *>       valid_resources;
	std::unordered_map<size_t, size_t> valid_lookup;        // uuid - index

	std::vector<std::unique_ptr<Resource<_Ty>>> deprecates;

	bool update = false;
};

struct ResourceManager::Impl
{
	std::map<ResourceType, std::unique_ptr<IResourceManager>> managers;
};

ResourceManager::ResourceManager(RHIContext *rhi_context)
{
	m_impl = new Impl;
	m_impl->managers.emplace(ResourceType::Texture2D, std::make_unique<TResourceManager<ResourceType::Texture2D>>(rhi_context));
	m_impl->managers.emplace(ResourceType::Mesh, std::make_unique<TResourceManager<ResourceType::Mesh>>(rhi_context));
	m_impl->managers.emplace(ResourceType::SkinnedMesh, std::make_unique<TResourceManager<ResourceType::SkinnedMesh>>(rhi_context));
	m_impl->managers.emplace(ResourceType::Material, std::make_unique<TResourceManager<ResourceType::Material>>(rhi_context));
	m_impl->managers.emplace(ResourceType::Animation, std::make_unique<TResourceManager<ResourceType::Animation>>(rhi_context));
	m_impl->managers.emplace(ResourceType::Prefab, std::make_unique<TResourceManager<ResourceType::Prefab>>(rhi_context));
	m_impl->managers.emplace(ResourceType::RenderPipeline, std::make_unique<TResourceManager<ResourceType::RenderPipeline>>(rhi_context));
	m_impl->managers.emplace(ResourceType::Scene, std::make_unique<TResourceManager<ResourceType::Scene>>(rhi_context));

	std::unordered_map<ResourceType, std::function<void(RHIContext *, const std::string &)>> loading_meta = {
#define LOADING_META(RESOURCE_TYPE)                                                                                     \
	{                                                                                                                   \
		RESOURCE_TYPE, [&](RHIContext *rhi_context, const std::string &name) { Add<RESOURCE_TYPE>(rhi_context, name); } \
	}

	    LOADING_META(ResourceType::Texture2D),
	    LOADING_META(ResourceType::Mesh),
	    LOADING_META(ResourceType::SkinnedMesh),
	    LOADING_META(ResourceType::Texture2D),
	    LOADING_META(ResourceType::Material),
	    LOADING_META(ResourceType::Animation),
	    LOADING_META(ResourceType::Prefab),
	    LOADING_META(ResourceType::RenderPipeline),
	    LOADING_META(ResourceType::Scene),
	};

	for (const auto &file : std::filesystem::directory_iterator("Asset/Meta/"))
	{
		std::string filename = file.path().filename().string();
		if (Path::GetInstance().GetFileExtension(filename) == ".asset")
		{
			std::string  resource_name = "";
			ResourceType resource_type = ResourceType::Unknown;

			size_t last_pos        = filename.find_last_of('.');
			size_t second_last_pos = filename.substr(0, last_pos).find_last_of('.');

			resource_name = filename.substr(0, second_last_pos);
			resource_type = (ResourceType) (std::atoi(filename.substr(second_last_pos + 1, last_pos - second_last_pos).c_str()));

			loading_meta.at(resource_type)(rhi_context, resource_name);
		}
	}
}

ResourceManager::~ResourceManager()
{
	delete m_impl;
}

void ResourceManager::Tick()
{
	for (auto &[type, manager] : m_impl->managers)
	{
		manager->Tick();
	}
}

IResource *ResourceManager::Get(ResourceType type, size_t uuid)
{
	return m_impl->managers.at(type)->Get(uuid);
}

RHITexture *ResourceManager::GetThumbnail(ResourceType type, size_t uuid)
{
	return m_impl->managers.at(type)->GetThumbnail(uuid);
}

bool ResourceManager::Valid(ResourceType type, size_t uuid)
{
	return m_impl->managers.at(type)->Valid(uuid);
}

bool ResourceManager::Has(ResourceType type, size_t uuid)
{
	return m_impl->managers.at(type)->Has(uuid);
}

size_t ResourceManager::Index(ResourceType type, size_t uuid)
{
	return m_impl->managers.at(type)->Index(uuid);
}

void ResourceManager::Import(ResourceType type, const std::string &path)
{
	m_impl->managers.at(type)->Import(this, path);
}

void ResourceManager::Erase(ResourceType type, size_t uuid)
{
	std::string asset_path = fmt::format("Asset/Meta/{}.{}.asset", m_impl->managers.at(type)->GetName(uuid), (uint32_t) type);
	if (Path::GetInstance().IsExist(asset_path))
	{
		Path::GetInstance().DeletePath(asset_path);
	}
	m_impl->managers.at(type)->Erase(uuid);
}

void ResourceManager::Add(ResourceType type, std::unique_ptr<IResource> &&resource)
{
	size_t uuid = resource->GetUUID();
	m_impl->managers.at(type)->Add(std::move(resource), uuid);
}

const std::vector<std::string> ResourceManager::GetResources(ResourceType type, bool only_valid) const
{
	return m_impl->managers.at(type)->GetResources(only_valid);
}

bool ResourceManager::Update(ResourceType type) const
{
	return m_impl->managers.at(type)->Update();
}

void ResourceManager::SetDirty(ResourceType type)
{
	return m_impl->managers.at(type)->SetDirty();
}
}        // namespace Ilum