#pragma once

#include "Graphics/Image/Image.hpp"
#include "Math/Vector3.h"

namespace Ilum::Cmpt
{
enum class LightType
{
	Directional,
	Point,
	Spot
	// TODO: LTC
};

struct Light
{
	// Shadow mapping
	Image                shadow_map;
	std::array<Image, 6> shadow_cube;

	// Light type
	LightType type = LightType::Directional;
};
}        // namespace Ilum::Cmpt