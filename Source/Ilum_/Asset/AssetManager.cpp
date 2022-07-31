#include "AssetManager.hpp"
#include "Material.hpp"

#include <Core/Path.hpp>

#include <RHI/ImGuiContext.hpp>

#pragma warning(push, 0)
#define CGLTF_IMPLEMENTATION
#include <cgltf/cgltf.h>
#pragma warning(pop)

#include <imgui.h>

#include <IconsFontAwesome4.h>

namespace Ilum
{
AssetManager::AssetManager(RHIDevice *device) :
    p_device(device)
{
}

Texture *AssetManager::LoadTexture(const std::string &filename)
{
	m_textures.emplace_back(std::make_unique<Texture>(p_device, filename));
	m_texture_lookup.emplace(m_textures.back().get(), static_cast<uint32_t>(m_textures.size()) - 1);
	m_textures.back()->SetName(Path::GetInstance().GetFileName(filename, false));
	m_update_texture = true;
	return m_textures.back().get();
}

Texture *AssetManager::Add(std::unique_ptr<Texture> &&texture)
{
	m_textures.emplace_back(std::move(texture));
	m_texture_lookup.emplace(m_textures.back().get(), static_cast<uint32_t>(m_textures.size()) - 1);
	m_update_texture = true;
	return m_textures.back().get();
}

Material *AssetManager::Add(std::unique_ptr<Material> &&material)
{
	m_materials.emplace_back(std::move(material));
	m_material_lookup.emplace(m_materials.back().get(), static_cast<uint32_t>(m_materials.size()) - 1);
	m_update_material = true;
	return m_materials.back().get();
}

Mesh *AssetManager::Add(std::unique_ptr<Mesh> &&mesh)
{
	m_meshes.emplace_back(std::move(mesh));
	m_mesh_lookup.emplace(m_meshes.back().get(), static_cast<uint32_t>(m_meshes.size()) - 1);
	m_update_mesh = true;
	return m_meshes.back().get();
}

void AssetManager::Erase(Texture *texture)
{
	if (m_texture_lookup.find(texture) != m_texture_lookup.end())
	{
		uint32_t                 deprecated_index = m_texture_lookup[texture];
		std::unique_ptr<Texture> deprecated       = std::move(m_textures[deprecated_index]);
		m_textures[deprecated_index]              = std::move(m_textures.back());
		m_textures.pop_back();
		m_texture_lookup.erase(deprecated.get());
		if (deprecated_index < m_textures.size())
		{
			m_texture_lookup[m_textures[deprecated_index].get()] = deprecated_index;
			m_update_texture                                     = true;
		}
	}
}

void AssetManager::Erase(Mesh *mesh)
{
	if (m_mesh_lookup.find(mesh) != m_mesh_lookup.end())
	{
		uint32_t              deprecated_index = m_mesh_lookup[mesh];
		std::unique_ptr<Mesh> deprecated       = std::move(m_meshes[deprecated_index]);
		m_meshes[deprecated_index]             = std::move(m_meshes.back());
		m_meshes.pop_back();
		m_mesh_lookup.erase(deprecated.get());
		if (deprecated_index < m_meshes.size())
		{
			m_mesh_lookup[m_meshes[deprecated_index].get()] = deprecated_index;
			m_update_mesh                                   = true;
		}
	}
}

void AssetManager::Erase(Material *material)
{
	if (m_material_lookup.find(material) != m_material_lookup.end())
	{
		uint32_t                  deprecated_index = m_material_lookup[material];
		std::unique_ptr<Material> deprecated       = std::move(m_materials[deprecated_index]);
		m_materials[deprecated_index]              = std::move(m_materials.back());
		m_materials.pop_back();
		m_material_lookup.erase(deprecated.get());
		if (deprecated_index < m_materials.size())
		{
			m_material_lookup[m_materials[deprecated_index].get()] = deprecated_index;
			m_update_material                                      = true;
		}
	}
}

bool AssetManager::IsValid(Texture *texture)
{
	return m_texture_lookup.find(texture) != m_texture_lookup.end();
}

bool AssetManager::IsValid(Mesh *mesh)
{
	return m_mesh_lookup.find(mesh) != m_mesh_lookup.end();
}

bool AssetManager::IsValid(Material *material)
{
	return m_material_lookup.find(material) != m_material_lookup.end();
}

uint32_t AssetManager::GetIndex(Mesh *mesh)
{
	if (m_mesh_lookup.find(mesh) != m_mesh_lookup.end())
	{
		return m_mesh_lookup[mesh];
	}
	return ~0U;
}

uint32_t AssetManager::GetIndex(Texture *texture)
{
	if (m_texture_lookup.find(texture) != m_texture_lookup.end())
	{
		return m_texture_lookup[texture];
	}
	return ~0U;
}

uint32_t AssetManager::GetIndex(Material *material)
{
	if (m_material_lookup.find(material) != m_material_lookup.end())
	{
		return m_material_lookup[material];
	}
	return ~0U;
}

Mesh *AssetManager::GetMesh(uint32_t index)
{
	if (m_meshes.size() <= index)
	{
		return nullptr;
	}
	return m_meshes[index].get();
}

Texture *AssetManager::GetTexture(uint32_t index)
{
	if (m_textures.size() <= index)
	{
		return nullptr;
	}
	return m_textures[index].get();
}

Material *AssetManager::GetMaterial(uint32_t index)
{
	if (m_materials.size() <= index)
	{
		return nullptr;
	}
	return m_materials[index].get();
}

const std::vector<Buffer *> &AssetManager::GetVertexBuffer()
{
	return m_vertex_buffer;
}

const std::vector<Buffer *> &AssetManager::GetIndexBuffer()
{
	return m_index_buffer;
}

const std::vector<Buffer *> &AssetManager::GetMeshletVertexBuffer()
{
	return m_meshlet_vertex_buffer;
}

const std::vector<Buffer *> &AssetManager::GetMeshletTriangleBuffer()
{
	return m_meshlet_triangle_buffer;
}

const std::vector<Buffer *> &AssetManager::GetMeshletBuffer()
{
	return m_meshlet_buffer;
}

const std::vector<VkImageView> &AssetManager::GetTextureViews()
{
	return m_texture_views;
}

const std::vector<Buffer *> &AssetManager::GetMaterialBuffer()
{
	return m_material_buffer;
}

void AssetManager::Clear()
{
	m_meshes.clear();
	m_textures.clear();
	m_materials.clear();

	m_mesh_lookup.clear();
	m_texture_lookup.clear();
	m_material_lookup.clear();
}

bool AssetManager::OnImGui(ImGuiContext &context)
{
	bool is_update = false;

	ImGui::Begin("Asset Manager");

	// Textures
	if (ImGui::TreeNode("Texture"))
	{
		if (ImGui::BeginPopupContextItem("Texture##1"))
		{
			if (ImGui::MenuItem("Load"))
			{
				context.OpenFileDialog("Load Texture", "Load Texture", "Image file (*.png;*.jpg;*.jpeg;*.bmp;*.tga;*.hdr){.png,.jpg,.jpeg,.bmp,.tga,.hdr}");
			}
			ImGui::EndPopup();
		}

		int32_t tex_id = 0;
		for (auto &texture : m_textures)
		{
			if (!texture)
			{
				continue;
			}
			ImGui::PushID(tex_id++);
			bool open = ImGui::TreeNode(texture->GetName().c_str());
			// Drag & Drop
			if (ImGui::BeginDragDropSource())
			{
				uint32_t texture_index = GetIndex(texture.get());
				ImGui::SetDragDropPayload("Texture", &texture_index, sizeof(uint32_t));
				ImGui::EndDragDropSource();
			}
			// Popup window for delete
			if (ImGui::BeginPopupContextItem(texture->GetName().c_str()))
			{
				if (ImGui::MenuItem("Delete"))
				{
					Erase(texture.get());
					is_update = true;
				}
				ImGui::EndPopup();
			}
			// Display image
			if (open)
			{
				if (texture)
				{
					TextureViewDesc view_desc  = {};
					view_desc.aspect           = VK_IMAGE_ASPECT_COLOR_BIT;
					view_desc.view_type        = VK_IMAGE_VIEW_TYPE_2D;
					view_desc.base_array_layer = 0;
					view_desc.base_mip_level   = 0;
					view_desc.layer_count      = texture->GetLayerCount();
					view_desc.level_count      = texture->GetMipLevels();
					ImGui::Image(context.TextureID(texture->GetView(view_desc)), ImVec2(200, static_cast<float>(texture->GetHeight()) / static_cast<float>(texture->GetWidth()) * 200.f));
					ImGui::BulletText("Size: %ld x %ld", texture->GetWidth(), texture->GetHeight());
					ImGui::BulletText("Mips: %ld", texture->GetMipLevels());
					ImGui::BulletText("Layers: %ld", texture->GetLayerCount());
				}
				ImGui::TreePop();
			}
			ImGui::PopID();
		}
		ImGui::TreePop();
	}

	if (ImGui::TreeNode("Mesh"))
	{
		int32_t mesh_id = 0;
		for (auto &mesh : m_meshes)
		{
			ImGui::PushID(mesh_id++);
			bool open = ImGui::TreeNode(mesh->GetName().c_str());
			// Drag & Drop
			if (ImGui::BeginDragDropSource())
			{
				uint32_t mesh_index = GetIndex(mesh.get());
				ImGui::SetDragDropPayload("Mesh", &mesh_index, sizeof(uint32_t));
				ImGui::EndDragDropSource();
			}
			// Popup window for delete
			if (ImGui::BeginPopupContextItem(mesh->GetName().c_str()))
			{
				if (ImGui::MenuItem("Delete"))
				{
					Erase(mesh.get());
					is_update = true;
				}
				ImGui::EndPopup();
			}
			// Display Mesh
			if (open)
			{
				if (mesh->OnImGui(context))
				{
					is_update = true;
				}
				ImGui::TreePop();
			}
			ImGui::PopID();
		}
		ImGui::TreePop();
	}

	if (ImGui::TreeNode("Material"))
	{
		if (ImGui::BeginPopupContextItem("Material##1"))
		{
			if (ImGui::MenuItem("New"))
			{
				Add(std::make_unique<Material>(p_device, *this));
			}
			ImGui::EndPopup();
		}

		int32_t material_id = 0;
		for (auto &material : m_materials)
		{
			ImGui::PushID(material_id++);
			bool open = ImGui::TreeNode(material->GetName().c_str());
			// Drag & Drop
			if (ImGui::BeginDragDropSource())
			{
				uint32_t material_index = GetIndex(material.get());
				ImGui::SetDragDropPayload("Material", &material_index, sizeof(uint32_t));
				ImGui::EndDragDropSource();
			}
			// Popup window for delete
			if (ImGui::BeginPopupContextItem(material->GetName().c_str()))
			{
				if (ImGui::MenuItem("Delete"))
				{
					Erase(material.get());
				}
				ImGui::EndPopup();
			}
			// Display Material
			if (open)
			{
				if (material)
				{
					if (material->OnImGui(context))
					{
						is_update = true;
					}
				}
				ImGui::TreePop();
			}
			ImGui::PopID();
		}

		///////////////////////////////////////////////////
		for (auto& material : m_materials)
		{
			ImGui::PushID(material_id++);
			if (ImGui::Button(material->GetName().c_str()))
			{
				material->material_graph.EnableEditor();
			}
			material->material_graph.OnImGui(context);
			ImGui::PopID();
		}


		ImGui::TreePop();
	}

	context.GetFileDialogResult("Load Texture", [this, &is_update](const std::string &name) { LoadTexture(name); is_update=true; });

	ImGui::End();
	return is_update;
}
void AssetManager::Tick()
{
	if (m_update_mesh)
	{
		m_vertex_buffer.clear();
		m_index_buffer.clear();
		m_meshlet_vertex_buffer.clear();
		m_meshlet_triangle_buffer.clear();
		m_meshlet_buffer.clear();
		for (auto &mesh : m_meshes)
		{
			m_vertex_buffer.push_back(&mesh->GetVertexBuffer());
			m_index_buffer.push_back(&mesh->GetIndexBuffer());
			m_meshlet_vertex_buffer.push_back(&mesh->GetMeshletVertexBuffer());
			m_meshlet_triangle_buffer.push_back(&mesh->GetMeshletTriangleBuffer());
			m_meshlet_buffer.push_back(&mesh->GetMeshletBuffer());
		}
		m_update_mesh = false;
	}
	if (m_update_texture)
	{
		m_texture_views.clear();
		for (auto &texture : m_textures)
		{
			m_texture_views.push_back(texture->GetView(TextureViewDesc{
			    VK_IMAGE_VIEW_TYPE_2D,
			    VK_IMAGE_ASPECT_COLOR_BIT,
			    0,
			    texture->GetMipLevels(),
			    0,
			    texture->GetLayerCount()}));
		}
		m_update_texture = false;
	}
	if (m_update_material)
	{
		m_material_buffer.clear();
		for (auto &material : m_materials)
		{
			m_material_buffer.push_back(&material->GetBuffer());
		}
		m_update_material = false;
	}
}
}        // namespace Ilum