#pragma once

#include <glm/glm.hpp>

namespace Ilum::cmpt
{
	struct Transform
	{
	    glm::vec3 translation = {0.f, 0.f, 0.f};
	    glm::vec3 rotation = {0.f, 0.f, 0.f};
	    glm::vec3 scale = {1.f, 1.f, 1.f};

		glm::mat4 local_transform = glm::mat4(1.f);
	    glm::mat4 world_transform = glm::mat4(1.f);

		bool update;
	};
}