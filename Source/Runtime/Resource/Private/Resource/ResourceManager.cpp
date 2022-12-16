#include "ResourceManager.hpp"
#include "Importer.hpp"
#include "Resource.hpp"
#include "Resource/Animation.hpp"
#include "Resource/Mesh.hpp"
#include "Resource/Prefab.hpp"
#include "Resource/SkinnedMesh.hpp"
#include "Resource/Texture.hpp"

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

	virtual bool Has(size_t uuid) = 0;

	virtual size_t Index(size_t uuid) = 0;

	virtual void Erase(size_t uuid) = 0;

	virtual void Import(ResourceManager *manager, const std::string &path) = 0;

	virtual void Add(std::unique_ptr<IResource> &&resource, size_t uuid) = 0;

	virtual const std::vector<std::string> GetResources() const = 0;

	virtual bool Update() const = 0;

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
		if (lookup.find(uuid) != lookup.end())
		{
			return resources.at(lookup.at(uuid)).get();
		}
		return nullptr;
	}

	virtual bool Has(size_t uuid) override
	{
		return lookup.find(uuid) != lookup.end();
	}

	virtual size_t Index(size_t uuid) override
	{
		return Has(uuid) ? lookup.at(uuid) : ~0U;
	}

	virtual void Erase(size_t uuid) override
	{
		if (lookup.find(uuid) != lookup.end())
		{
			size_t last_uuid = ~0U;
			if (resources.size() > 1)
			{
				for (auto &[uuid_, index] : lookup)
				{
					if (index + 1 == lookup.size())
					{
						last_uuid = uuid_;
						break;
					}
				}

				lookup[last_uuid] = lookup[uuid];
				std::swap(resources.back(), resources[lookup[uuid]]);
			}

			deprecates.emplace_back(std::move(resources.back()));
			resources.pop_back();
			lookup.erase(uuid);
			update = true;
		}
	}

	virtual void Import(ResourceManager *manager, const std::string &path) override
	{
		Importer<_Ty>::Import(manager, path, rhi_context);
	}

	virtual void Add(std::unique_ptr<IResource> &&resource, size_t uuid) override
	{
		if (lookup.find(uuid) == lookup.end())
		{
			lookup.emplace(uuid, resources.size());
			resources.emplace_back(std::unique_ptr<Resource<_Ty>>(dynamic_cast<Resource<_Ty> *>(resource.release())));
			update = true;
		}
	}

	virtual const std::vector<std::string> GetResources() const override
	{
		std::vector<std::string> handles(resources.size());
		std::transform(resources.begin(), resources.end(), handles.begin(), [](const auto &resource) {
			return resource->GetName();
		});
		return handles;
	}

	virtual bool Update() const override
	{
		return update;
	}

	std::vector<std::unique_ptr<Resource<_Ty>>> resources;
	std::unordered_map<size_t, size_t>          lookup;        // uuid - index
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
	m_impl->managers.emplace(ResourceType::Animation, std::make_unique<TResourceManager<ResourceType::Animation>>(rhi_context));
	m_impl->managers.emplace(ResourceType::Prefab, std::make_unique<TResourceManager<ResourceType::Prefab>>(rhi_context));
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
	m_impl->managers.at(type)->Erase(uuid);
}

void ResourceManager::Add(ResourceType type, std::unique_ptr<IResource> &&resource)
{
	size_t uuid = resource->GetUUID();
	m_impl->managers.at(type)->Add(std::move(resource), uuid);
}

const std::vector<std::string> ResourceManager::GetResources(ResourceType type) const
{
	return m_impl->managers.at(type)->GetResources();
}

bool ResourceManager::Update(ResourceType type) const
{
	return m_impl->managers.at(type)->Update();
}
}        // namespace Ilum