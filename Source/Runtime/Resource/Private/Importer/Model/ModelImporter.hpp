#pragma once

//#include "Importer/Texture/TextureImporter.hpp"
#include "Resource/ResourceMeta.hpp"

#include <vector>

namespace Ilum
{
STRUCT(ModelImportInfo, Enable)
{
	std::string name;

	std::vector<Submesh> submeshes;

	std::vector<Vertex>   vertices;
	std::vector<uint32_t> indices;

	std::vector<Meshlet>  meshlets;
	std::vector<uint32_t> meshlet_vertices;
	std::vector<uint32_t> meshlet_primitives;

	AABB aabb;

	// TODO: Material

	//std::vector<TextureImportInfo> textures;
};

class ModelImporter
{
  public:
	virtual ModelImportInfo ImportImpl(const std::string &filename) = 0;

	static ModelImportInfo Import(const std::string &filename);

  protected:
	uint32_t PackTriangle(uint8_t v0, uint8_t v1, uint8_t v2);
};

}        // namespace Ilum