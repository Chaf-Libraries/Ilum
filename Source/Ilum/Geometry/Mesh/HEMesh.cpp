#include "HEMesh.hpp"

namespace Ilum::geometry
{
HEMesh::HEMesh(const std::vector<glm::vec3> &vertices, const std::vector<uint32_t> &indices, uint32_t stride) :
    m_stride(stride)
{
	// Create new vertices
	for (auto &v : vertices)
	{
		Vertex *nv = (Vertex *) m_pool.allocate(sizeof(Vertex), alignof(Vertex));
		new (nv) Vertex(v, nullptr);

		m_vertices.insert(nv);
	}

	std::unordered_map<std::pair<Vertex *, Vertex *>, HalfEdge *, pair_hash> vertex_edge_map;

	// Create new edge
	for (size_t i = 0; i < indices.size(); i += stride)
	{
		Face *nf = (Face *) m_pool.allocate(sizeof(Face), alignof(Face));
		new (nf) Face(nullptr);

		m_faces.insert(nf);

		std::vector<HalfEdge *> fe;

		for (size_t j = 0; j < stride; j++)
		{
			Vertex *v1 = m_vertices[indices[i + j]];
			Vertex *v2 = m_vertices[indices[i + (j + 1) % stride]];

			if (vertex_edge_map.find(std::make_pair(v1, v2)) == vertex_edge_map.end())
			{
				HalfEdge *ne = (HalfEdge *) m_pool.allocate(sizeof(HalfEdge), alignof(HalfEdge));
				new (ne) HalfEdge(v2, nf, nullptr, nullptr, nullptr);

				v1->half_edge = ne;

				if (vertex_edge_map.find(std::make_pair(v2, v1)) != vertex_edge_map.end())
				{
					auto *opposite     = vertex_edge_map[std::make_pair(v2, v1)];
					ne->opposite       = opposite;
					opposite->opposite = ne;
				}

				vertex_edge_map[std::make_pair(v1, v2)] = ne;

				m_halfedges.insert(ne);
				fe.push_back(ne);
			}
			else
			{
				auto *ne = vertex_edge_map[std::make_pair(v1, v2)];
				fe.push_back(ne);
			}
		}

		nf->half_edge = fe[0];

		for (size_t i = 0; i < fe.size(); i++)
		{
			fe[i]->next = fe[(i + 1) % fe.size()];
			fe[i]->prev = fe[(i + fe.size() - 1) % fe.size()];
		}
	}
}

HEMesh::~HEMesh()
{
	m_pool.release();
	m_vertices.clear();
	m_faces.clear();
	m_halfedges.clear();
}

const std::vector<HEMesh::Vertex *> &HEMesh::vertices() const
{
	return m_vertices.vec();
}

const std::vector<HEMesh::Face *> &HEMesh::faces() const
{
	return m_faces.vec();
}

const std::vector<HEMesh::HalfEdge *> &HEMesh::halfEdges() const
{
	return m_halfedges.vec();
}

std::pair<std::vector<glm::vec3>, std::vector<uint32_t>> HEMesh::toMesh() const
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
		auto *  he      = f->half_edge;
		Vertex *start_v = he->prev->vertex;
		he              = f->half_edge->next;

		while (he != f->half_edge)
		{
			indices.push_back(static_cast<uint32_t>(m_vertices.idx(start_v)));
			indices.push_back(static_cast<uint32_t>(m_vertices.idx(he->vertex)));
			indices.push_back(static_cast<uint32_t>(m_vertices.idx(he->next->vertex)));

			he = he->next;
		}
	}

	return std::make_pair(vertices, indices);
}
}        // namespace Ilum::geometry