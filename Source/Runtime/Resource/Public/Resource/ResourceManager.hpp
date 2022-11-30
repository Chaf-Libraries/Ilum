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
		return Has(Get(Type, uuid));
	}

	template <ResourceType Type>
	size_t Index(size_t uuid)
	{
		return Index(Type, uuid);
	}

	template <ResourceType Type>
	Resource<Type> *Import(const std::string& importer, const std::string &path)
	{
		return static_cast<Resource<Type> *>(Import(Type, importer, path));
	}

  private:
	IResource *Get(ResourceType type, size_t uuid);

	bool Has(ResourceType type, size_t uuid);

	size_t Index(ResourceType type, size_t uuid);

	IResource *Import(ResourceType type, const std::string &importer, const std::string &path);

  private:
	struct Impl;
	Impl *m_impl = nullptr;
};
}        // namespace Ilum