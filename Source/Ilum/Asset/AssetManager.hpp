#pragma once

#include "Mesh.hpp"

#include <RHI/Device.hpp>
#include <RHI/Texture.hpp>

namespace Ilum
{
class ImGuiContext;
class Material;

class AssetManager
{
  public:
	AssetManager(RHIDevice *device);
	~AssetManager() = default;

	Texture *LoadTexture(const std::string &filename);

	Texture  *Add(std::unique_ptr<Texture> &&texture);
	Material *Add(std::unique_ptr<Material> &&material);
	Mesh     *Add(std::unique_ptr<Mesh> &&mesh);

	void Erase(Texture *texture);
	void Erase(Mesh *mesh);
	void Erase(Material *material);

	bool IsValid(Texture *texture);
	bool IsValid(Mesh *mesh);
	bool IsValid(Material *material);

	uint32_t GetIndex(Mesh *mesh);
	uint32_t GetIndex(Texture *texture);
	uint32_t GetIndex(Material *material);

	Mesh     *GetMesh(uint32_t index);
	Texture  *GetTexture(uint32_t index);
	Material *GetMaterial(uint32_t index);

	const std::vector<Buffer *> &GetVertexBuffer();
	const std::vector<Buffer *> &GetIndexBuffer();
	const std::vector<Buffer *> &GetMeshletVertexBuffer();
	const std::vector<Buffer *> &GetMeshletTriangleBuffer();
	const std::vector<Buffer *> &GetMeshletBuffer();
	const std::vector<VkImageView> &GetTextureViews();
	const std::vector<Buffer *> &GetMaterialBuffer();

	void Clear();

	bool OnImGui(ImGuiContext &context);

	template <class Archive>
	void serialize(Archive &ar)
	{
		ar(m_meshes);
	}

	void Tick();

  private:
	RHIDevice *p_device = nullptr;

	std::vector<std::unique_ptr<Mesh>>     m_meshes;
	std::vector<std::unique_ptr<Texture>>  m_textures;
	std::vector<std::unique_ptr<Material>> m_materials;

	std::unordered_map<Mesh *, uint32_t>     m_mesh_lookup;
	std::unordered_map<Texture *, uint32_t>  m_texture_lookup;
	std::unordered_map<Material *, uint32_t> m_material_lookup;

	bool m_update_mesh     = false;
	bool m_update_texture  = false;
	bool m_update_material = false;

	std::vector<Buffer *>  m_vertex_buffer;
	std::vector<Buffer *>  m_index_buffer;
	std::vector<Buffer *>  m_meshlet_vertex_buffer;
	std::vector<Buffer *>  m_meshlet_triangle_buffer;
	std::vector<Buffer *>  m_meshlet_buffer;
	std::vector<VkImageView> m_texture_views;
	std::vector<Buffer *>  m_material_buffer;
};
}        // namespace Ilum