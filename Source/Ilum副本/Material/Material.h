#pragma once

#include "Utils/PCH.hpp"

namespace Ilum
{
struct Material
{
	virtual std::type_index type() = 0;

	Material()
	{
		update = true;
	}

	~Material()
	{
		update = true;
	}

	inline static bool update = false;
};

template <typename T>
struct TMaterial : public Material
{
	virtual std::type_index type() override
	{
		return typeid(T);
	}
};

using MaterialReference = std::reference_wrapper<scope<Material>>;
}        // namespace Ilum::material