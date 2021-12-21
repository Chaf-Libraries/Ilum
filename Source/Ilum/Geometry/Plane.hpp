#pragma once

#include <glm/glm.hpp>

namespace Ilum::geometry
{
	struct Plane
	{
		// Plane equation: Ax+By+Cz+D=0
	    glm::vec3 normal   = glm::vec3(0.f);
	    float     constant = 0;

		Plane() = default;

		Plane(const glm::vec3 &p0, const glm::vec3 &p1, const glm::vec3 &p2);

		Plane(const glm::vec3 &normal, const glm::vec3 &point);

		Plane(const glm::vec3 &normal, float distance);

		~Plane() = default;

		Plane transform(const glm::mat4 &trans) const;

		glm::vec3 reflect(const glm::vec3 &direction) const;

		float distance(const glm::vec3 &p) const;
	};
}