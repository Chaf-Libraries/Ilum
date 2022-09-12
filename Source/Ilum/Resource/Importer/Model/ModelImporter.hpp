#pragma once

#include "Resource/Importer/Texture/TextureImporter.hpp"

#include <Geometry/Bound/AABB.hpp>
#include <Geometry/Vertex.hpp>

#include <meshoptimizer.h>

#include <vector>

namespace Ilum
{
struct Meshlet
{
	struct Bound
	{
		std::array<float, 3> center;
		float                radius;

		std::array<float, 3> cone_axis;
		float                cone_cutoff;
		alignas(16) std::array<float, 3> cone_apex;

		template <class Archive>
		void serialize(Archive &ar)
		{
			ar(center, radius, cone_apex, cone_cutoff, cone_apex);
		}
	} bound;

	uint32_t indices_offset;
	uint32_t indices_count;
	uint32_t vertices_offset;        // Global offset
	uint32_t vertices_count;
	uint32_t meshlet_vertices_offset;        // Meshlet offset
	uint32_t meshlet_indices_offset;         // Meshlet offset

	template <class Archive>
	void serialize(Archive &ar)
	{
		ar(bound, indices_offset, indices_count, vertices_offset, vertices_count, meshlet_vertices_offset, meshlet_indices_offset);
	}
};

struct Submesh
{
	std::string name;

	uint32_t index;

	glm::mat4 pre_transform;

	AABB aabb;

	uint32_t vertices_count;
	uint32_t vertices_offset;
	uint32_t indices_count;
	uint32_t indices_offset;
	uint32_t meshlet_count;
	uint32_t meshlet_offset;
	// TODO: Material

	template <class Archive>
	void serialize(Archive &ar)
	{
		ar(name, index, pre_transform, aabb, vertices_count, vertices_offset, indices_count, indices_offset, meshlet_count, meshlet_offset);
	}
};

struct ModelImportInfo
{
	std::string name;

	std::vector<Submesh> submeshes;

	std::vector<Vertex>   vertices;
	std::vector<uint32_t> indices;

	std::vector<Meshlet>  meshlets;
	std::vector<uint32_t> meshlet_vertices;
	std::vector<uint8_t>  meshlet_indices;

	AABB aabb;

	// TODO: Material

	std::vector<TextureImportInfo> textures;

	template <class Archive>
	void serialize(Archive &ar)
	{
		ar(submeshes, vertices, indices, meshlets, meshlet_vertices, meshlet_indices, aabb, textures);
	}
};

class ModelImporter
{
  public:
	virtual ModelImportInfo ImportImpl(const std::string &filename) = 0;

	static ModelImportInfo Import(const std::string &filename);
};

}        // namespace Ilum