#pragma once

#include <glm/glm.hpp>

#include "RandomSet.hpp"

namespace Ilum::geometry
{
// Face based data structure only support triangle mesh
class FMesh
{
  public:
	struct Face;
	struct Vertex;

	struct Vertex
	{
		glm::vec3 position;
		Face *    face;

		Vertex(const glm::vec3 &position, Face *face) :
		    position(position), face(face)
		{}
	};

	struct Face
	{
		Vertex *vertex[3];
		Face *  face[3];

		Face(Vertex *v1, Vertex *v2, Vertex *v3, Face *f1, Face *f2, Face *f3) :
		    vertex{v1, v2, v3}, face{f1, f2, f3}
		{}
	};

  public:
	FMesh(const std::vector<glm::vec3> &vertices, const std::vector<uint32_t> &indices);

	~FMesh();

	const std::vector<Vertex *> &vertices() const;

	const std::vector<Face *> &faces() const;

  private:
	std::pmr::unsynchronized_pool_resource m_pool;

	RandomSet<Vertex *> m_vertices;
	RandomSet<Face *>   m_faces;
};
}        // namespace Ilum::geometry