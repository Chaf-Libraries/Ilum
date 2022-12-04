#pragma once

#include "Mesh.hpp"

namespace Ilum
{
class EXPORT_API HEMesh : public Mesh
{
  public:
	struct Vertex;
	struct Face;
	struct HalfEdge;

	struct Vertex
	{
		VertexData data;
		HalfEdge  *half_edge;

		Vertex(const VertexData &data, HalfEdge *half_edge) :
		    data(data), half_edge(half_edge)
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
		Vertex   *vertex;
		Face     *face;
		HalfEdge *next;
		HalfEdge *prev;
		HalfEdge *opposite;

		HalfEdge(Vertex *vertex, Face *face, HalfEdge *next, HalfEdge *prev, HalfEdge *opposite) :
		    vertex(vertex), face(face), next(next), prev(prev), opposite(opposite)
		{}
	};

	HEMesh(const std::vector<VertexData> &vertices, const std::vector<uint32_t> &indices, uint32_t stride = 3);

	~HEMesh();

	const std::vector<Vertex *> &Vertices() const;

	const std::vector<Face *> &Faces() const;

	const std::vector<HalfEdge *> &HalfEdges() const;

	std::vector<Vertex *> &Vertices();

	std::vector<Face *> &Faces();

	std::vector<HalfEdge *> &HalfEdges();

	bool HasBoundary() const;

	bool IsOnBoundary(HalfEdge *he) const;

	bool IsOnBoundary(Vertex *v) const;

	uint32_t Degree(Vertex *v) const;

	std::vector<std::vector<Vertex *>> Boundary() const;

	std::vector<Vertex *> AdjVertices(Vertex *v) const;

	size_t VertexIndex(Vertex *v) const;

	virtual TriMesh ToTriMesh() const override;

  private:
	uint32_t m_stride;

	RandomSet<Vertex *>   m_vertices;
	RandomSet<HalfEdge *> m_halfedges;
	RandomSet<Face *>     m_faces;
};
}        // namespace Ilum