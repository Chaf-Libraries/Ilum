#pragma once

#include "Component.hpp"

#include <glm/glm.hpp>

namespace Ilum
{
STRUCT(TransformComponent, Enable) :
    public Component
{
	glm::vec3 translation = {0.f, 0.f, 0.f};
	glm::vec3 rotation    = {0.f, 0.f, 0.f};
	glm::vec3 scale       = {1.f, 1.f, 1.f};

	glm::mat4 local_transform = glm::mat4(1.f);
	glm::mat4 world_transform = glm::mat4(1.f);
};
}        // namespace Ilum