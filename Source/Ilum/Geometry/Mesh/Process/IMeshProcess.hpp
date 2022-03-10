#pragma once

#include "Geometry/Vertex.hpp"

#include <vector>

#include <glm/glm.hpp>

namespace Ilum::geometry
{
class IMeshProcess
{
  public:
	static const std::vector<glm::vec3> preprocess(const std::vector<Vertex> &vertices);

	static const std::vector<Vertex> postprocess(const std::vector<glm::vec3> &vertices, const std::vector<uint32_t> &indices, const std::vector<glm::vec2> &texcoords = {});
};
}        // namespace Ilum::geometry