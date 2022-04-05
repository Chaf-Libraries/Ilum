#pragma once

#include <glm/glm.hpp>

#include <cereal/cereal.hpp>

namespace Ilum
{
struct Vertex
{
	alignas(16) glm::vec3 position  = {};
	alignas(16) glm::vec2 texcoord  = {};
	alignas(16) glm::vec3 normal    = {};

	Vertex() = default;

	Vertex(glm::vec3 position, glm::vec2 texcoord, glm::vec3 normal) :
	    position(position),
	    texcoord(texcoord),
	    normal(normal)
	{
	}

	template<class Archive>
	void serialize(Archive& ar)
	{
		ar(position.x, position.y, position.z,
		   texcoord.x, texcoord.y,
		   normal.x, normal.y, normal.z);
	}
};
}        // namespace Ilum