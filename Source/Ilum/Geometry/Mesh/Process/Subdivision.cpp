#include "Subdivision.hpp"

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/hash.hpp>

#include <tbb/tbb.h>

namespace Ilum::geometry
{
inline std::unordered_set<uint32_t> find_boundary(const std::vector<glm::vec3> &vertices, const std::vector<uint32_t> &indices)
{
	std::vector<std::vector<size_t>> mesh_graph(vertices.size(), std::vector<size_t>(vertices.size()));
	for (auto &g : mesh_graph)
	{
		std::fill(g.begin(), g.end(), 0);
	}

	for (size_t i = 0; i < indices.size(); i += 3)
	{
		mesh_graph[indices[i]][indices[i + 1]]++;
		mesh_graph[indices[i + 1]][indices[i]]++;
		mesh_graph[indices[i + 1]][indices[i + 2]]++;
		mesh_graph[indices[i + 2]][indices[i + 1]]++;
		mesh_graph[indices[i + 2]][indices[i]]++;
		mesh_graph[indices[i]][indices[i + 2]]++;
	}

	std::unordered_set<size_t>   found;
	std::unordered_set<uint32_t> boundary_points;

	for (size_t i = 0; i < vertices.size(); i++)
	{
		size_t currect_vertex = i;
		while (true)
		{
			bool has = false;
			for (size_t j = 0; j < vertices.size(); j++)
			{
				if (j != currect_vertex && mesh_graph[currect_vertex][j] == 1 && found.find(j) == found.end())
				{
					found.insert(j);
					boundary_points.insert(static_cast<uint32_t>(j));
					currect_vertex = j;
					has            = true;
				}
			}
			if (currect_vertex == i || !has)
			{
				break;
			}
		}
	}

	return boundary_points;
}

std::pair<std::vector<Vertex>, std::vector<uint32_t>> Subdivision::LoopSubdivision(const std::vector<Vertex> &in_vertices, const std::vector<uint32_t> &in_indices)
{
	//std::vector<glm::vec3> vertices = preprocess(in_vertices);
	//std::vector<uint32_t>  indices  = in_indices;

	std::vector<glm::vec3> raw_vertices = preprocess(in_vertices);
	std::vector<uint32_t>  raw_indices  = in_indices;

	std::vector<glm::vec3> vertices;
	std::vector<uint32_t>  indices;
	std::vector<glm::vec2> texcoords;

	//optimize mesh
	std::unordered_map<glm::vec3, uint32_t> vertices_map;
	for (uint32_t i = 0; i < raw_vertices.size(); i++)
	{
		vertices_map[raw_vertices[i]] = i;
	}

	for (auto &[key, index] : vertices_map)
	{
		texcoords.push_back(in_vertices[index].texcoord);
		index       = static_cast<uint32_t>(vertices.size());
		glm::vec3 v = {};
		v           = key;
		vertices.push_back(v);
	}

	for (auto &idx : raw_indices)
	{
		indices.push_back(vertices_map[raw_vertices[idx]]);
	}

	uint32_t old_vertices_count = static_cast<uint32_t>(vertices.size());
	uint32_t old_triangle_count = static_cast<uint32_t>(indices.size() / 3);

	std::map<uint32_t, std::unordered_set<uint32_t>>                       vertex_connect_map;
	std::map<uint32_t, std::vector<uint32_t>>                              triangle_connect_map;
	std::unordered_map<std::pair<uint32_t, uint32_t>, uint32_t, pair_hash> new_vertex_map;

	for (uint32_t i = 0; i < indices.size(); i += 3)
	{
		for (uint32_t j = 0; j < 3; j++)
		{
			// Adding vertex connection
			vertex_connect_map[indices[i + j]].insert(indices[i + (j + 1) % 3]);
			vertex_connect_map[indices[i + (j + 1) % 3]].insert(indices[i + j]);

			// Add triangle connection
			triangle_connect_map[indices[i + j]].push_back(i / 3);

			// Adding new vertices
			uint32_t nv = 0;
			if (new_vertex_map.find(std::make_pair(indices[i + j], indices[i + (j + 1) % 3])) == new_vertex_map.end())
			{
				glm::vec3 v = (vertices[indices[i + j]] + vertices[indices[i + (j + 1) % 3]]) * 0.5f;
				vertices.push_back(v);

				glm::vec2 uv = (texcoords[indices[i + j]] + texcoords[indices[i + (j + 1) % 3]]) * 0.5f;
				texcoords.push_back(uv);

				new_vertex_map[std::make_pair(indices[i + j], indices[i + (j + 1) % 3])] = static_cast<uint32_t>(vertices.size() - 1);
				new_vertex_map[std::make_pair(indices[i + (j + 1) % 3], indices[i + j])] = static_cast<uint32_t>(vertices.size() - 1);

				nv = static_cast<uint32_t>(vertices.size() - 1);
			}
			else
			{
				nv = new_vertex_map[std::make_pair(indices[i + j], indices[i + (j + 1) % 3])];
			}

			vertex_connect_map[nv].insert(indices[i + j]);
			vertex_connect_map[nv].insert(indices[i + (j + 1) % 3]);
			triangle_connect_map[nv].push_back(i / 3);
		}
	}

	// Reconstruct mesh
	std::vector<uint32_t> new_indices;
	for (uint32_t i = 0; i < indices.size(); i += 3)
	{
		uint32_t v0 = indices[i];
		uint32_t v1 = indices[i + 1];
		uint32_t v2 = indices[i + 2];

		uint32_t nv0 = new_vertex_map[std::make_pair(v0, v1)];
		uint32_t nv1 = new_vertex_map[std::make_pair(v1, v2)];
		uint32_t nv2 = new_vertex_map[std::make_pair(v2, v0)];

		std::vector<uint32_t> triangle_indices = {
		    v0, nv0, nv2,
		    nv0, v1, nv1,
		    nv2, nv1, v2,
		    nv1, nv2, nv0};

		new_indices.insert(new_indices.end(), triangle_indices.begin(), triangle_indices.end());
	}

	//auto boundary_points = find_boundary(vertices, new_indices);
	std::unordered_set<uint32_t> boundary_points;

	// Update new vertex position
	// 3/8 *(A+B)+1/8*(C+D)
	//				D
	//           /   \
		//         A-n-B
	//			 \    /
	//            C
	std::vector<glm::vec3> new_vertices(vertices.size());
	std::vector<glm::vec2> new_texcoords(vertices.size());

	boundary_points.reserve(vertices.size());
	tbb::parallel_for(tbb::blocked_range<uint32_t>(old_vertices_count, static_cast<uint32_t>(vertices.size())), [&](const tbb::blocked_range<uint32_t> &count) {
		for (uint32_t i = count.begin(); i != count.end(); i++)
		{
			glm::vec3 new_pos = {};
			glm::vec2 new_uv  = {};

			// Check boundary
			if (triangle_connect_map[i].size() == 1)
			{
				for (auto v : vertex_connect_map[i])
				{
					boundary_points.insert(v);
					new_pos += 0.5f * vertices[v];
					new_uv += 0.5f * texcoords[v];
				}
			}
			else
			{
				for (auto t : triangle_connect_map[i])
				{
					for (uint32_t j = 0; j < 3; j++)
					{
						new_pos += 0.125f * vertices[indices[3 * t + j]];
						new_uv += 0.125f * texcoords[indices[3 * t + j]];
					}
				}

				for (auto v : vertex_connect_map[i])
				{
					new_pos += 0.125f * vertices[v];
					new_uv += 0.125f * texcoords[v];
				}
			}

			new_vertices[i]  = new_pos;
			new_texcoords[i] = new_uv;
		}
	});

	// Update old vertex position
	tbb::parallel_for(tbb::blocked_range<uint32_t>(0, old_vertices_count), [&](const tbb::blocked_range<uint32_t> &count) {
		for (uint32_t i = count.begin(); i != count.end(); i++)
		{
			glm::vec3 new_pos = {};
			glm::vec2 new_uv  = {};

			if (boundary_points.find(i) == boundary_points.end())
			{
				uint32_t degree = static_cast<uint32_t>(vertex_connect_map[i].size());
				float    u      = 0.f;
				if (degree == 3)
				{
					u = 3.f / 16.f;
				}
				else
				{
					u = 3.f / (8.f * static_cast<float>(degree));
				}

				for (auto v : vertex_connect_map[i])
				{
					new_pos += u * vertices[v];
					new_uv += u * texcoords[v];
				}
				new_pos += (1 - static_cast<float>(degree) * u) * vertices[i];
				new_uv += (1 - static_cast<float>(degree) * u) * texcoords[i];
			}
			else
			{
				new_pos = vertices[i];
				new_uv  = texcoords[i];
			}

			new_vertices[i]  = new_pos;
			new_texcoords[i] = new_uv;
		}
	});

	return std::make_pair(std::move(postprocess(new_vertices, new_indices, new_texcoords)), std::move(new_indices));
}
}        // namespace Ilum::geometry