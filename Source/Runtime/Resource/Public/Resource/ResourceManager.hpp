#pragma once

#include "Resource.hpp"

#include <RHI/RHIContext.hpp>

namespace Ilum
{
class EXPORT_API ResourceManager
{
  public:
	ResourceManager(RHIContext *rhi_context);

	~ResourceManager();

	template <ResourceType Type>
	Resource<Type> *Get(size_t uuid)
	{
		return static_cast<Resource<Type> *>(Get(Type, uuid));
	}

	template <ResourceType Type>
	bool Has(size_t uuid)
	{
		return Has(Type, uuid);
	}

	template <ResourceType Type>
	size_t Index(size_t uuid)
	{
		return Index(Type, uuid);
	}

	template <ResourceType Type>
	void Import(const std::string &path)
	{
		 Import(Type, path);
	}

	template<ResourceType Type>
	void Add(std::unique_ptr<Resource<Type>>&& resource, size_t uuid)
	{
		Add(Type, std::move(resource), uuid);
	}

	template <ResourceType Type>
	const std::vector<size_t> GetResources() const
	{
		return GetResources(Type);
	}

  private:
	IResource *Get(ResourceType type, size_t uuid);

	bool Has(ResourceType type, size_t uuid);

	size_t Index(ResourceType type, size_t uuid);

	void Import(ResourceType type, const std::string &path);

	void Add(ResourceType type, std::unique_ptr<IResource> &&resource, size_t uuid);

	const std::vector<size_t> GetResources(ResourceType type) const;

  private:
	struct Impl;
	Impl *m_impl = nullptr;
};
}        // namespace Ilum