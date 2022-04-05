#include "IMeshProcess.hpp"

namespace Ilum::geometry
{
const std::vector<glm::vec3> IMeshProcess::preprocess(const std::vector<Vertex> &vertices)
{
	std::vector<glm::vec3> result(vertices.size());
	for (size_t i = 0; i < vertices.size(); i++)
	{
		result[i] = vertices[i].position;
	}
	return result;
}

const std::vector<Vertex> IMeshProcess::postprocess(const std::vector<glm::vec3> &vertices, const std::vector<uint32_t> &indices, const std::vector<glm::vec2> &texcoords)
{
	std::vector<Vertex> result(vertices.size());

	for (size_t i = 0; i < indices.size(); i += 3)
	{
		glm::vec3 e2 = vertices[indices[i + 1]] - vertices[indices[i]];
		glm::vec3 e1 = vertices[indices[i + 2]] - vertices[indices[i]];

		glm::vec3 normal = glm::normalize(glm::cross(e1, e2));
		result[indices[i]].normal += normal;
		result[indices[i + 1]].normal += normal;
		result[indices[i + 2]].normal += normal;
	}

	for (size_t i = 0; i < vertices.size(); i++)
	{
		result[i].position = vertices[i];
		result[i].normal   = glm::normalize(result[i].normal);
		if (!texcoords.empty())
		{
			result[i].texcoord = texcoords[i];
		}
	}

	return result;
}
}        // namespace Ilum::geometry