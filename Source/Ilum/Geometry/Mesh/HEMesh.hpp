#pragma once

#include <glm/glm.hpp>

#include "RandomSet.hpp"

namespace Ilum::geometry
{
class HEMesh
{
  public:
	struct Vertex;
	struct Face;
	struct HalfEdge;

	struct Vertex
	{
		glm::vec3 position;
		HalfEdge *half_edge;

		Vertex(const glm::vec3 &position, HalfEdge *half_edge) :
		    position(position), half_edge(half_edge)
		{}
	};

	struct Face
	{
		HalfEdge *half_edge;

		Face(HalfEdge *half_edge) :
		    half_edge(half_edge)
		{}
	};

	struct HalfEdge
	{
		Vertex *  vertex;
		Face *    face;
		HalfEdge *next;
		HalfEdge *prev;
		HalfEdge *opposite;

		HalfEdge(Vertex *vertex, Face *face, HalfEdge *next, HalfEdge *prev, HalfEdge *opposite) :
		    vertex(vertex), face(face), next(next), prev(prev), opposite(opposite)
		{}
	};

	HEMesh(const std::vector<glm::vec3> &vertices, const std::vector<uint32_t> &indices, uint32_t stride = 3);
	~HEMesh();

	const std::vector<Vertex *> &vertices() const;

	const std::vector<Face *> &faces() const;

	const std::vector<HalfEdge *> &halfEdges() const;

	std::pair<std::vector<glm::vec3>, std::vector<uint32_t>> toMesh() const;

  private:
	uint32_t m_stride;

	std::pmr::unsynchronized_pool_resource m_pool;

	RandomSet<Vertex *>   m_vertices;
	RandomSet<HalfEdge *> m_halfedges;
	RandomSet<Face *>     m_faces;
};
}        // namespace Ilum::geometry