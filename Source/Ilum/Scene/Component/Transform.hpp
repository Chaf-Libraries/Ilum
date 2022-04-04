#pragma once

#include "ComponentSerializer.hpp"

#include <glm/glm.hpp>

#include <cereal/cereal.hpp>

namespace Ilum::cmpt
{
struct Transform
{
	glm::vec3 translation = {0.f, 0.f, 0.f};
	glm::vec3 rotation    = {0.f, 0.f, 0.f};
	glm::vec3 scale       = {1.f, 1.f, 1.f};

	glm::mat4 local_transform = glm::mat4(1.f);
	glm::mat4 world_transform = glm::mat4(1.f);

	inline static bool update = false;

	template <class Archive>
	void serialize(Archive &ar)
	{
		glm::serialize(ar, translation);
		glm::serialize(ar, rotation);
		glm::serialize(ar, scale);
	}
};
}        // namespace Ilum::cmpt