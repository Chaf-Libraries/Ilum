#include <Geometry/MeshProcess.hpp>

using namespace Ilum;

class LoopSubdivision : public Subdivision
{
  public:
	virtual TriMesh Execute(const TriMesh &mesh) override
	{
		std::vector<VertexData> vertices = mesh.vertices;
		std::vector<uint32_t>   indices  = mesh.indices;

		uint32_t old_vertices_count = static_cast<uint32_t>(vertices.size());
		uint32_t old_triangle_count = static_cast<uint32_t>(indices.size() / 3);

		std::map<uint32_t, std::unordered_set<uint32_t>>                      vertex_connect_map;
		std::map<uint32_t, std::vector<uint32_t>>                             triangle_connect_map;
		std::unordered_map<std::pair<uint32_t, uint32_t>, uint32_t, PairHash> new_vertex_map;

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
					VertexData v;
					v.position = (vertices[indices[i + j]].position + vertices[indices[i + (j + 1) % 3]].position) * 0.5f;
					v.uv       = (vertices[indices[i + j]].uv + vertices[indices[i + (j + 1) % 3]].uv) * 0.5f;
					vertices.push_back(v);

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

		// auto boundary_points = find_boundary(vertices, new_indices);
		std::unordered_set<uint32_t> boundary_points;

		// Update new vertex position
		// 3/8 *(A+B)+1/8*(C+D)
		//				D
		//           /   \
		//         A-n-B
		//			 \    /
		//            C
		std::vector<VertexData> new_vertices(vertices.size());

		boundary_points.reserve(vertices.size());
		for (uint32_t i = old_vertices_count; i < vertices.size(); i++)
		{
			VertexData new_vertex = {};

			// Check boundary
			if (triangle_connect_map[i].size() == 1)
			{
				for (auto v : vertex_connect_map[i])
				{
					boundary_points.insert(v);
					new_vertex.position += 0.5f * vertices[v].position;
					new_vertex.uv += 0.5f * vertices[v].uv;
				}
			}
			else
			{
				for (auto t : triangle_connect_map[i])
				{
					for (uint32_t j = 0; j < 3; j++)
					{
						new_vertex.position += 0.125f * vertices[indices[3 * t + j]].position;
						new_vertex.uv += 0.125f * vertices[indices[3 * t + j]].uv;
					}
				}

				for (auto v : vertex_connect_map[i])
				{
					new_vertex.position += 0.125f * vertices[v].position;
					new_vertex.uv += 0.125f * vertices[v].uv;
				}
			}

			new_vertices[i] = new_vertex;
		}

		// Update old vertex position
		for (uint32_t i = 0; i < old_vertices_count; i++)
		{
			VertexData new_v = {};

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
					new_v.position += u * vertices[v].position;
					new_v.uv += u * vertices[v].uv;
				}
				new_v.position += (1 - static_cast<float>(degree) * u) * vertices[i].position;
				new_v.uv += (1 - static_cast<float>(degree) * u) * vertices[i].uv;
			}
			else
			{
				new_v.position = vertices[i].position;
				new_v.uv       = vertices[i].uv;
			}

			new_vertices[i] = new_v;
		}

		TriMesh result;
		result.vertices = std::move(new_vertices);
		result.indices  = std::move(new_indices);
		result.GenerateNormal();
		return result;
	}
};

extern "C"
{
	EXPORT_API LoopSubdivision *Create()
	{
		return new LoopSubdivision;
	}
}