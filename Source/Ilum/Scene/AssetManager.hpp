#pragma once

#include "Mesh.hpp"

#include <RHI/Device.hpp>
#include <RHI/Texture.hpp>

namespace Ilum
{
class ImGuiContext;

class AssetManager
{
  public:
	AssetManager(RHIDevice *device);
	~AssetManager() = default;

	Mesh    *LoadMesh(const std::string &filename);
	Texture *LoadTexture(const std::string &filename);
	
	Material *AddMaterial(std::unique_ptr<Material> &&material);

	uint32_t GetMeshIndex(Mesh *mesh);
	uint32_t GetTextureIndex(Texture *texture);
	uint32_t GetMaterialIndex(Material *material);

	bool OnImGui(ImGuiContext& context);

  private:
	RHIDevice *p_device = nullptr;

	std::vector<std::unique_ptr<Mesh>>     m_meshes;
	std::vector<std::unique_ptr<Texture>>  m_textures;
	std::vector<std::unique_ptr<Material>> m_materials;

	std::unordered_map<Mesh *, uint32_t> m_mesh_lookup;
	std::unordered_map<Texture *, uint32_t> m_texture_lookup;
	std::unordered_map<Material *, uint32_t> m_material_lookup;
};
}        // namespace Ilum