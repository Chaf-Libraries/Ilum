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

			// Not find v1 -> v2, create new one
			if (vertex_edge_map.find(std::make_pair(v1, v2)) == vertex_edge_map.end())
			{
				HalfEdge *ne = (HalfEdge *) m_pool.allocate(sizeof(HalfEdge), alignof(HalfEdge));
				new (ne) HalfEdge(v2, nf, nullptr, nullptr, nullptr);

				v1->half_edge = ne;

				vertex_edge_map[std::make_pair(v1, v2)] = ne;
				m_halfedges.insert(ne);
				fe.push_back(ne);

				if (vertex_edge_map.find(std::make_pair(v2, v1)) != vertex_edge_map.end())
				{
					auto *opposite     = vertex_edge_map[std::make_pair(v2, v1)];
					ne->opposite       = opposite;
					opposite->opposite = ne;
				}
				else
				{
					HalfEdge *opposite = (HalfEdge *) m_pool.allocate(sizeof(HalfEdge), alignof(HalfEdge));
					new (opposite) HalfEdge(v1, nullptr, nullptr, nullptr, nullptr);
					ne->opposite       = opposite;
					opposite->opposite = ne;

					vertex_edge_map[std::make_pair(v2, v1)] = opposite;
					m_halfedges.insert(opposite);
				}
			}
			else
			{
				auto *ne = vertex_edge_map[std::make_pair(v1, v2)];
				ne->face = nf;
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

	for (auto *he : m_halfedges)
	{
		if (!he->face)
		{
			he->next = he->opposite->prev->opposite;
			while (he->next->face)
			{
				he->next = he->next->prev->opposite;
			}

			he->prev = he->opposite->next->opposite;
			while (he->prev->face)
			{
				he->prev = he->prev->next->opposite;
			}
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

std::vector<HEMesh::Vertex *> &HEMesh::vertices()
{
	return m_vertices.vec();
}

std::vector<HEMesh::Face *> &HEMesh::faces()
{
	return m_faces.vec();
}

std::vector<HEMesh::HalfEdge *> &HEMesh::halfEdges()
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
		auto *he = f->half_edge;
		auto *h  = he;

		do
		{
			indices.push_back(static_cast<uint32_t>(m_vertices.idx(he->vertex)));
			he = he->next;
		} while (h != he);
	}

	return std::make_pair(vertices, indices);
}

bool HEMesh::hasBoundary() const
{
	for (auto &he : m_halfedges)
	{
		if (!he->face)
		{
			return true;
		}
	}

	return false;
}

bool HEMesh::onBoundary(HalfEdge *he) const
{
	return !he->face || !he->opposite->face;
}

bool HEMesh::onBoundary(Vertex *v) const
{
	auto *he = v->half_edge;
	auto *h  = he;
	do
	{
		if (h->opposite->face)
		{
			h = h->opposite->next;
		}
		else
		{
			return true;
		}
	} while (he != h);

	return false;
}

uint32_t HEMesh::degree(Vertex *v) const
{
	auto *he = v->half_edge;
	auto *h  = he;

	uint32_t deg = 0;

	do
	{
		deg++;
		h = h->opposite->next;
	} while (h != he);

	return deg;
}

std::vector<std::vector<HEMesh::Vertex *>> HEMesh::boundary() const
{
	std::unordered_set<HEMesh::HalfEdge *>     boundary_halfedges;
	std::vector<std::vector<HEMesh::Vertex *>> boundaries;

	for (auto *he : m_halfedges)
	{
		if (!he->face && boundary_halfedges.find(he) == boundary_halfedges.end())
		{
			std::vector<HEMesh::Vertex *> boundary;
			auto *                        h = he;

			do
			{
				boundary_halfedges.insert(h);
				boundary.push_back(h->vertex);
				h = h->next;
			} while (h != he);

			boundaries.push_back(boundary);
		}
	}

	return boundaries;
}

std::vector<HEMesh::Vertex *> HEMesh::adjVertices(Vertex *v) const
{
	auto *he = v->half_edge;
	auto *h  = he;

	std::vector<HEMesh::Vertex *> adj_vertices;

	do
	{
		adj_vertices.push_back(h->vertex);
		h = h->opposite->next;
	} while (h != he);

	return adj_vertices;
}

size_t HEMesh::vertexIndex(Vertex *v) const
{
	return m_vertices.idx(v);
}
}        // namespace Ilum::geometry