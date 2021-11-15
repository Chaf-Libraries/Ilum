#pragma once

#include "Light.hpp"

#include <glm/glm.hpp>

namespace Ilum::cmpt
{
struct DirectionalLight: public TLight<LightType::Directional>
{
	float intensity = 1.f;
	glm::vec3 color     = {1.f, 1.f, 1.f};
	glm::vec3 direction;
};
}