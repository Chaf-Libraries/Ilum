#pragma once

#include "Utils/PCH.hpp"

namespace Ilum
{
struct IMaterial
{
	virtual std::type_index type() = 0;
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