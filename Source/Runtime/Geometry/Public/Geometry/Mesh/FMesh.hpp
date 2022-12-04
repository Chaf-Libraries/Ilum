#pragma once

#include "Mesh.hpp"

namespace Ilum
{
// Face based data structure only support triangle mesh
class EXPORT_API FMesh : public Mesh
{
  public:
	struct Face;
	struct Vertex;

	struct Vertex
	{
		VertexData data;
		Face      *face;

		Vertex(const VertexData &data, Face *face) :
		    data(data), face(face)
		{}
	};

	struct Face
	{
		Vertex *vertex[3];
		Face   *face[3];

		Face(Vertex *v1, Vertex *v2, Vertex *v3, Face *f1, Face *f2, Face *f3) :
		    vertex{v1, v2, v3}, face{f1, f2, f3}
		{}
	};

  public:
	FMesh(const std::vector<VertexData> &vertices, const std::vector<uint32_t> &indices);

	~FMesh();

	virtual TriMesh ToTriMesh() const override;

  private:
	RandomSet<Vertex *> m_vertices;
	RandomSet<Face *>   m_faces;
};
}        // namespace Ilum