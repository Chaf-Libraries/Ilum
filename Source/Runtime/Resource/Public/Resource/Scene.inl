#pragma once

#include "Resource.hpp"

namespace Ilum
{
template <>
class TResource<ResourceType::Scene> : public Resource
{
  public:
	explicit TResource(size_t uuid, const std::string &meta) :
	    Resource(uuid, meta)
	{
	}
	int a;

  private:
};
}        // namespace Ilum