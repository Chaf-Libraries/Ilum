#include "FMesh.hpp"

#include <tbb/tbb.h>

namespace Ilum::geometry
{
FMesh::FMesh(const std::vector<glm::vec3> &vertices, const std::vector<uint32_t> &indices)
{
	// Create new vertices
	for (auto &v : vertices)
	{
		Vertex *nv = (Vertex *) m_pool.allocate(sizeof(Vertex), alignof(Vertex));
		new (nv) Vertex(v, nullptr);

		m_vertices.insert(nv);
	}

	std::unordered_map<std::pair<Vertex *, Vertex *>, Face *, pair_hash> edge_face_map;
	std::unordered_map<Face *, uint32_t>                                 face_count;

	for (size_t idx = 0; idx < indices.size(); idx += 3)
	{
		Face *nf = (Face *) m_pool.allocate(sizeof(Face), alignof(Face));
		new (nf) Face(m_vertices[indices[idx]], m_vertices[indices[idx + 1]], m_vertices[indices[idx + 2]], nullptr, nullptr, nullptr);

		m_faces.insert(nf);

		face_count[nf] = 0;

		for (size_t i = 0; i < 3; i++)
		{
			if (!m_vertices[indices[idx + i]]->face)
			{
				m_vertices[indices[idx + i]]->face = nf;
			}

			auto e1 = std::make_pair(nf->vertex[i], nf->vertex[(i + 1) % 3]);
			auto e2 = std::make_pair(nf->vertex[(i + 1) % 3], nf->vertex[i]);

			if (edge_face_map.find(e1) != edge_face_map.end())
			{
				nf->face[face_count[nf]++]                               = edge_face_map[e1];
				edge_face_map[e1]->face[face_count[edge_face_map[e1]]++] = nf;
			}

			edge_face_map[e1] = nf;
			edge_face_map[e2] = nf;
		}
	}
}

FMesh::~FMesh()
{
	m_pool.release();
	m_vertices.clear();
	m_faces.clear();
}

const std::vector<FMesh::Vertex *> &FMesh::vertices() const
{
	return m_vertices.vec();
}

const std::vector<FMesh::Face *> &FMesh::faces() const
{
	return m_faces.vec();
}

std::pair<std::vector<glm::vec3>, std::vector<uint32_t>> FMesh::toMesh() const
{
	std::vector<glm::vec3> vertices;
	std::vector<uint32_t>  indices;

	vertices.reserve(m_vertices.size());
	for (auto& v : m_vertices)
	{
		vertices.push_back(v->position);
	}

	indices.reserve(m_faces.size() * 3);
	for (auto& f : m_faces)
	{
		for (uint32_t i = 0; i < 3; i++)
		{
			indices.push_back(static_cast<uint32_t>(m_vertices.idx(f->vertex[i])));
		}
	}

	return std::make_pair(vertices, indices);
}
}        // namespace Ilum::geometry