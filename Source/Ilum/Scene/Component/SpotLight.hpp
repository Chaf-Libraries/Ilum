#pragma once

#include "Light.hpp"

#include <glm/glm.hpp>

namespace Ilum::cmpt
{
struct SpotLight : public TLight<LightType::Spot>
{
	struct Data
	{
		glm::vec3 color         = {1.f, 1.f, 1.f};
		float     intensity     = 1.f;
		glm::vec3 position      = {0.f, 0.f, 0.f};
		float     cut_off       = glm::cos(glm::radians(12.5f));
		glm::vec3 direction     = {1.f, 1.f, 1.f};
		float     outer_cut_off = glm::cos(glm::radians(17.5f));
	}data;
};
}        // namespace Ilum::cmpt