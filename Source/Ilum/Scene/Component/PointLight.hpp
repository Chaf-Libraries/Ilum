#pragma once

#include "Light.hpp"

namespace Ilum::cmpt
{
struct PointLight : public TLight<LightType::Point>
{
	struct Data
	{
		glm::vec3 color     = {1.f, 1.f, 1.f};
		float     intensity = 1.f;
		glm::vec3 position  = {0.f, 0.f, 0.f};
		float     constant  = 1.0f;
		alignas(16) float linear = 0.09f;
		 float quadratic = 0.032f;
	}data;
};
}