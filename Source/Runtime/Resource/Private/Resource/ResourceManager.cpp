#include "ResourceManager.hpp"
#include "Resource.hpp"

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

	virtual IResource *Has(size_t uuid) = 0;

	virtual size_t Index(size_t uuid) = 0;

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

		return nullptr;
	}

	virtual Resource<_Ty> *Has(size_t uuid) override;

	virtual size_t Index(size_t uuid) override;

	std::vector<std::unique_ptr<Resource<_Ty>>> resources;
	std::unordered_map<size_t, size_t> lookup;	// uuid - index
	std::vector<std::unique_ptr<Resource<_Ty>>> deprecates;
};

struct ResourceManager::Impl
{
	std::map<ResourceType, std::vector<IResource *>>
};

ResourceManager::ResourceManager(RHIContext *rhi_context)
{
	m_impl = new Impl;
}

ResourceManager::~ResourceManager()
{
	delete m_impl;
}

IResource *ResourceManager::Get(std::type_index index, size_t uuid)
{
	return nullptr;
}

bool ResourceManager::Has(std::type_index index, size_t uuid)
{
	return false;
}

size_t ResourceManager::Index(std::type_index index, size_t uuid)
{
	return size_t();
}

IResource *ResourceManager::Import(std::type_index index, size_t uuid)
{
	return nullptr;
}
}        // namespace Ilum