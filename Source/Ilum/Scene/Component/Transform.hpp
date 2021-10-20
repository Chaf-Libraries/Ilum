#pragma once

#include <glm/glm.hpp>

namespace Ilum::cmpt
{
	struct Transform
	{
	    glm::vec3 position = {0.f, 0.f, 0.f};
	    glm::vec3 rotation = {0.f, 0.f, 0.f};
	    glm::vec3 scale = {0.f, 0.f, 0.f};

		glm::mat4 local_transform;
	    glm::mat4 world_transform;

		bool update;
	};
}