#pragma once

#include "Mesh/Mesh.hpp"

namespace Ilum
{
// Edge based data structure
class EXPORT_API EMesh : public Mesh
{
  public:
	struct Vertex;
	struct Edge;
	struct Face;

	struct Vertex
	{
		VertexData data;
		Edge      *edge = nullptr;

		Vertex(const VertexData &data, Edge *edge) :
		    data(data), edge(edge)
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
		Face   *face[2];
		Edge   *next[2];
		Edge   *prev[2];

		Edge(Vertex *v1, Vertex *v2, Face *f1, Face *f2, Edge *next_e1, Edge *next_e2, Edge *prev_e1, Edge *prev_e2) :
		    vertex{v1, v2}, face{f1, f2}, next{next_e1, next_e2}, prev{prev_e1, prev_e2}
		{}
	};

  public:
	EMesh(const std::vector<VertexData> &vertices, const std::vector<uint32_t> &indices, uint32_t stride = 3);

	~EMesh();

	virtual TriMesh ToTriMesh() const override;

  private:
	uint32_t m_stride;

	RandomSet<Vertex *> m_vertices;
	RandomSet<Face *>   m_faces;
	RandomSet<Edge *>   m_edges;
};
}        // namespace Ilum