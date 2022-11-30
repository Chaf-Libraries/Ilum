#include "ResourceManager.hpp"
#include "Resource.hpp"
#include "Importer.hpp"
#include "Resource/Texture.hpp"
#include "Resource/Model.hpp"

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

	virtual IResource *Get(size_t uuid) = 0;

	virtual bool Has(size_t uuid) = 0;

	virtual size_t Index(size_t uuid) = 0;

	virtual void Erase(size_t uuid) = 0;

	virtual IResource *Import(const std::string &importer, const std::string &path) = 0;

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
		}

		deprecates.emplace_back(std::move(resources.back()));
		resources.pop_back();
		lookup.erase(uuid);
	}

	virtual IResource *Import(const std::string &importer, const std::string &path) override
	{
		if (lookup.find(Hash(path)) == lookup.end())
		{
			std::unique_ptr<Resource<_Ty>> resource = Importer<_Ty>::GetInstance(importer)->Import(path, rhi_context);
			lookup.emplace(Hash(path), resources.size());
			resources.emplace_back(std::move(resource));
			return resources.back().get();
		}
		return nullptr;
	}

	std::vector<std::unique_ptr<Resource<_Ty>>> resources;
	std::unordered_map<size_t, size_t>          lookup;        // uuid - index
	std::vector<std::unique_ptr<Resource<_Ty>>> deprecates;
};

struct ResourceManager::Impl
{
	std::map<ResourceType, std::unique_ptr<IResourceManager>> managers;
};

ResourceManager::ResourceManager(RHIContext *rhi_context)
{
	m_impl = new Impl;
	m_impl->managers.emplace(ResourceType::Texture, std::make_unique<TResourceManager<ResourceType::Texture>>(rhi_context));
	m_impl->managers.emplace(ResourceType::Model, std::make_unique<TResourceManager<ResourceType::Model>>(rhi_context));
}

ResourceManager::~ResourceManager()
{
	delete m_impl;
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

IResource *ResourceManager::Import(ResourceType type, const std::string &importer, const std::string &path)
{
	return m_impl->managers.at(type)->Import(importer, path);
}
}        // namespace Ilum