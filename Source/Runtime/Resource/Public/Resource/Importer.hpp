#pragma once

#include "Resource.hpp"

namespace Ilum
{
class RHIContext;
class ResourceManager;

template <ResourceType Type>
class Importer
{
  public:
	Importer() = default;

	virtual ~Importer() = default;

	static std::unique_ptr<Importer<Type>> &GetInstance(const std::string &plugin);

	static void Import(ResourceManager* manager, const std::string &path, RHIContext *rhi_context);

  protected:
	virtual void Import_(ResourceManager *manager, const std::string &path, RHIContext *rhi_context) = 0;
};
}        // namespace Ilum