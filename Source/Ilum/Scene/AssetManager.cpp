#include "AssetManager.hpp"

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

Mesh *AssetManager::LoadMesh(const std::string &filename)
{
	cgltf_options options{};
	cgltf_data   *raw_data = nullptr;
	cgltf_result  result   = cgltf_parse_file(&options, filename.c_str(), &raw_data);
	if (result != cgltf_result_success)
	{
		LOG_ERROR("Failed to load GLTF file: {}", filename);
	}
	result = cgltf_load_buffers(&options, raw_data, filename.c_str());
	if (result != cgltf_result_success)
	{
		LOG_ERROR("Failed to load GLTF's buffer: {}", filename);
	}


	std::map<const cgltf_image *, Texture *> texture_map;



	return nullptr;
}

Texture *AssetManager::LoadTexture(const std::string &filename)
{
	m_textures.emplace_back(std::make_unique<Texture>(p_device, filename));
	m_texture_lookup.emplace(m_textures.back().get(), static_cast<uint32_t>(m_textures.size()) - 1);
	m_textures.back()->SetName(Path::GetInstance().GetFileName(filename, false));
	return m_textures.back().get();
}

Material *AssetManager::AddMaterial(std::unique_ptr<Material> &&material)
{
	m_materials.emplace_back(std::move(material));
	m_material_lookup.emplace(m_materials.back().get(), static_cast<uint32_t>(m_materials.size()) - 1);
	return m_materials.back().get();
}

uint32_t AssetManager::GetMeshIndex(Mesh *mesh)
{
	if (m_mesh_lookup.find(mesh) != m_mesh_lookup.end())
	{
		return m_mesh_lookup[mesh];
	}
	return ~0U;
}

uint32_t AssetManager::GetTextureIndex(Texture *texture)
{
	if (m_texture_lookup.find(texture) != m_texture_lookup.end())
	{
		return m_texture_lookup[texture];
	}
	return ~0U;
}

uint32_t AssetManager::GetMaterialIndex(Material *material)
{
	if (m_material_lookup.find(material) != m_material_lookup.end())
	{
		return m_material_lookup[material];
	}
	return ~0U;
}

bool AssetManager::OnImGui(ImGuiContext &context)
{
	ImGui::Begin("Asset Manager");
	if (ImGui::Button("Load Texture"))
	{
		context.OpenFileDialog("Load Texture", "Load Texture", "Image file (*.png;*.jpg;*.jpeg;*.bmp;*.tga;*.hdr){.png,.jpg,.jpeg,.bmp,.tga,.hdr}");
	}

	// Textures
	if (ImGui::TreeNode("Texture"))
	{
		TextureViewDesc view_desc  = {};
		view_desc.aspect           = VK_IMAGE_ASPECT_COLOR_BIT;
		view_desc.view_type        = VK_IMAGE_VIEW_TYPE_2D;
		view_desc.base_array_layer = 0;
		view_desc.base_mip_level   = 0;
		view_desc.layer_count      = 1;
		view_desc.level_count      = 1;
		int32_t tex_id             = 0;
		for (auto &texture : m_textures)
		{
			ImGui::PushID(tex_id++);
			bool open = ImGui::TreeNode(texture->GetName().c_str());
			// Drag & Drop
			if (ImGui::BeginDragDropSource())
			{
				Texture *tex_ptr = texture.get();
				ImGui::SetDragDropPayload("Texture", &tex_ptr, sizeof(Texture *));
				ImGui::EndDragDropSource();
			}
			// Popup window for delete
			if (ImGui::BeginPopupContextItem(texture->GetName().c_str()))
			{
				if (ImGui::MenuItem("Delete"))
				{
					open = false;
					uint32_t id = m_texture_lookup[texture.get()];
					m_texture_lookup.erase(texture.get());
					m_textures.erase(m_textures.begin() + id);
				}
				ImGui::EndPopup();
			}
			// Display image
			if (open)
			{
				ImGui::Image(context.TextureID(texture->GetView(view_desc)), ImVec2(200, static_cast<float>(texture->GetHeight()) / static_cast<float>(texture->GetWidth()) * 200.f));
				ImGui::BulletText("Size: %ld x %ld", texture->GetWidth(), texture->GetHeight());
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
				Mesh *mesh_ptr = mesh.get();
				ImGui::SetDragDropPayload("Mesh", &mesh_ptr, sizeof(Mesh *));
				ImGui::EndDragDropSource();
			}
			// Popup window for delete
			if (ImGui::BeginPopupContextItem(mesh->GetName().c_str()))
			{
				if (ImGui::MenuItem("Delete"))
				{
					open            = false;
					uint32_t id = m_mesh_lookup[mesh.get()];
					m_mesh_lookup.erase(mesh.get());
					m_meshes.erase(m_meshes.begin() + id);
				}
				ImGui::EndPopup();
			}
			// Display image
			if (open)
			{

				ImGui::TreePop();
			}
			ImGui::PopID();
		}
		ImGui::TreePop();
	}

	context.GetFileDialogResult("Load Texture", [this](const std::string &name) { LoadTexture(name); });

	ImGui::End();
	return false;
}

}        // namespace Ilum