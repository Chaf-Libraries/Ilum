#pragma once

#include "Utils/PCH.hpp"

namespace Ilum
{
struct IMaterial
{
	virtual size_t size() = 0;

	virtual void *data() = 0;

	virtual std::type_index type() = 0;

	uint64_t id()
	{
		return (uint64_t) (this);
	}
};

template <typename T>
struct TMaterial : public IMaterial
{
	virtual std::type_index type() override
	{
		return typeid(T);
	}
};

using MaterialReference = std::reference_wrapper<scope<IMaterial>>;
}        // namespace Ilum::material