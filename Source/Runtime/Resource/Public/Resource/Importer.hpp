#pragma once

#include "Resource.hpp"

namespace Ilum
{
class RHIContext;

template <ResourceType Type>
class Importer
{
  public:
	Importer() = default;

	virtual ~Importer() = default;

	static std::unique_ptr<Importer<Type>> &GetInstance(const std::string &plugin);

	virtual std::unique_ptr<Resource<Type>> Import(const std::string &path, RHIContext *rhi_context) = 0;
};
}        // namespace Ilum