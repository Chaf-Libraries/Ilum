#pragma once

#include "Graphics/Image/Image.hpp"
#include <glm/glm.hpp>

namespace Ilum::cmpt
{
enum class LightType
{
	None,
	Directional,
	Point,
	Spot,
	Area
};

struct ILight
{
	virtual LightType type() = 0;
};

template <LightType Type>
struct TLight : public ILight
{
	virtual LightType type() override
	{
		return Type;
	}
};

struct Light
{
	// Light type
	LightType type = LightType::None;

	scope<ILight> impl = nullptr;

	inline static std::atomic<bool> update = false;
};
}        // namespace Ilum::Cmpt