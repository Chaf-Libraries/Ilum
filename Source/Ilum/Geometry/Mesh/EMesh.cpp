#include "EMesh.hpp"

namespace Ilum::geometry
{
EMesh::EMesh(const std::vector<glm::vec3> &vertices, const std::vector<uint32_t> &indices, uint32_t stride) :
    m_stride(stride)
{
	// Create new vertices
	for (auto &v : vertices)
	{
		Vertex *nv = (Vertex *) m_pool.allocate(sizeof(Vertex), alignof(Vertex));
		new (nv) Vertex(v, nullptr);

		m_vertices.insert(nv);
	}

	std::unordered_map<std::pair<Vertex *, Vertex *>, Edge *, pair_hash> vertex_edge_map;

	// Create new edge
	for (size_t i = 0; i < indices.size(); i += stride)
	{
		Face *nf = (Face *) m_pool.allocate(sizeof(Face), alignof(Face));
		new (nf) Face(nullptr);

		std::vector<Edge *> fe;

		for (size_t j = 0; j < stride; j++)
		{
			Vertex *v1 = m_vertices[indices[i + j]];
			Vertex *v2 = m_vertices[indices[i + (j + 1) % stride]];

			if (vertex_edge_map.find(std::make_pair(v1, v2)) == vertex_edge_map.end() &&
			    vertex_edge_map.find(std::make_pair(v2, v1)) == vertex_edge_map.end())
			{
				Edge *ne = (Edge *) m_pool.allocate(sizeof(Edge), alignof(Edge));
				new (ne) Edge(v1, v2, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);

				vertex_edge_map[std::make_pair(v1, v2)] = ne;
				vertex_edge_map[std::make_pair(v2, v1)] = ne;

				m_edges.insert(ne);
				fe.push_back(ne);
			}
			else
			{
				auto *ne = vertex_edge_map[std::make_pair(v1, v2)];
				fe.push_back(ne);
			}
		}

		nf->edge = fe[0];

		for (size_t j = 0; j < fe.size(); j++)
		{
			fe[j]->face[0] == nullptr ? fe[j]->face[0] = nf : fe[j]->face[1] = nf;

			Edge *e1 = fe[j];
			Edge *e2 = fe[(j + 1) % fe.size()];

			if (e1 && e2)
			{
				e2->prev[0] == nullptr ? e2->prev[0] = e1 : e2->prev[1] = e1;
				e1->next[0] == nullptr ? e1->next[0] = e2 : e1->next[1] = e2;
			}
		}

		m_faces.insert(nf);
	}
}

EMesh::~EMesh()
{
	m_pool.release();
	m_vertices.clear();
	m_faces.clear();
	m_edges.clear();
}

const std::vector<EMesh::Vertex *> &EMesh::vertices() const
{
	return m_vertices.vec();
}

const std::vector<EMesh::Face *> &EMesh::faces() const
{
	return m_faces.vec();
}

const std::vector<EMesh::Edge *> &EMesh::edges() const
{
	return m_edges.vec();
}

std::pair<std::vector<glm::vec3>, std::vector<uint32_t>> EMesh::toMesh() const
{
	std::vector<glm::vec3> vertices;
	std::vector<uint32_t>  indices;

	vertices.reserve(m_vertices.size());
	for (auto &v : m_vertices)
	{
		vertices.push_back(v->position);
	}

	// Convert to triangle mesh
	indices.reserve(m_faces.size() * (m_stride - 2) * 3);
	for (auto &f : m_faces)
	{
		Edge *   e        = f->edge;
		uint32_t face_idx = e->face[0] == f ? 0 : 1;
		Vertex * start_v  = e->vertex[face_idx];

		e = f->edge->next[face_idx];

		while (e->vertex[0] != start_v && e->vertex[1] != start_v)
		{
			indices.push_back(static_cast<uint32_t>(m_vertices.idx(start_v)));
			indices.push_back(static_cast<uint32_t>(m_vertices.idx(e->vertex[face_idx])));
			indices.push_back(static_cast<uint32_t>(m_vertices.idx(e->vertex[(face_idx + 1) % 2])));

			face_idx = e->face[0] == f ? 0 : 1;

			e = e->next[face_idx];
		}
	}

	return std::make_pair(vertices, indices);
}
}        // namespace Ilum::geometry