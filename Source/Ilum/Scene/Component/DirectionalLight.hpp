#pragma once

#include "Light.hpp"

#include <glm/glm.hpp>

namespace Ilum::cmpt
{
struct DirectionalLight : public TLight<LightType::Directional>
{
	struct Data
	{
		glm::vec3 color                 = {1.f, 1.f, 1.f};
		float     intensity             = 1.f;
		alignas(16) glm::vec3 direction = {1.f, 1.f, 1.f};
	} data;
};
}        // namespace Ilum::cmpt