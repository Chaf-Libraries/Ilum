#pragma once

#include <vector>

#include <glm/glm.hpp>

#include "RandomSet.hpp"

namespace Ilum::geometry
{
// Edge based data structure
class EMesh
{
  public:
	struct Vertex;
	struct Edge;
	struct Face;

	struct Vertex
	{
		glm::vec3 position;
		Edge *    edge = nullptr;

		Vertex(const glm::vec3 &position, Edge *edge) :
		    position(position), edge(edge)
		{}
	};

	struct Face
	{
		Edge *edge = nullptr;

		Face(Edge *edge) :
		    edge(edge)
		{}
	};

	struct Edge
	{
		Vertex *vertex[2];
		Face *  face[2];
		Edge *  next[2];
		Edge *  prev[2];

		Edge(Vertex *v1, Vertex *v2, Face *f1, Face *f2, Edge *next_e1, Edge *next_e2, Edge *prev_e1, Edge *prev_e2) :
		    vertex{v1, v2}, face{f1, f2}, next{next_e1, next_e2}, prev{prev_e1, prev_e2}
		{}
	};

  public:
	EMesh(const std::vector<glm::vec3> &vertices, const std::vector<uint32_t> &indices, uint32_t stride = 3);
	~EMesh();

	const std::vector<Vertex *> &vertices() const;

	const std::vector<Face *> &faces() const;

	const std::vector<Edge *> &edges() const;

	std::pair<std::vector<glm::vec3>, std::vector<uint32_t>> toMesh() const;

  private:
	uint32_t m_stride;

	std::pmr::unsynchronized_pool_resource m_pool;

	RandomSet<Vertex *> m_vertices;
	RandomSet<Face *>   m_faces;
	RandomSet<Edge *>   m_edges;
};
}        // namespace Ilum::geometry