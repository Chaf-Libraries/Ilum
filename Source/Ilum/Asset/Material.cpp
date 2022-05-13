#include "Material.hpp"
#include "AssetManager.hpp"

#include <imgui.h>

#include <glm/gtc/type_ptr.hpp>

namespace Ilum
{
inline bool DrawTextureButton(Texture *&texture, ImGuiContext &context, AssetManager &manager)
{
	bool is_update = false;

	if (texture && manager.IsValid(texture))
	{
		TextureViewDesc view_desc  = {};
		view_desc.aspect           = VK_IMAGE_ASPECT_COLOR_BIT;
		view_desc.view_type        = VK_IMAGE_VIEW_TYPE_2D;
		view_desc.base_array_layer = 0;
		view_desc.base_mip_level   = 0;
		view_desc.layer_count      = texture->GetLayerCount();
		view_desc.level_count      = texture->GetMipLevels();

		if (ImGui::ImageButton(context.TextureID(texture->GetView(view_desc)), ImVec2(200, 200)))
		{
			texture   = nullptr;
			is_update = true;
		}
	}
	else
	{
		ImGui::ImageButton(ImGui::GetIO().Fonts->TexID, ImVec2(200, 200), ImVec2(0, 0), ImVec2(0, 0));
	}
	if (ImGui::BeginDragDropTarget())
	{
		if (const auto *pay_load = ImGui::AcceptDragDropPayload("Texture"))
		{
			ASSERT(pay_load->DataSize == sizeof(uint32_t));
			if (manager.GetIndex(texture) != *static_cast<uint32_t *>(pay_load->Data))
			{
				texture   = manager.GetTexture(*static_cast<uint32_t *>(pay_load->Data));
				is_update = true;
			}
		}
		ImGui::EndDragDropTarget();
	}
	return is_update;
}

Material::Material(RHIDevice *device, AssetManager &manager) :
    p_device(device), m_manager(manager)
{
	BufferDesc desc   = {};
	desc.size         = sizeof(ShaderInterop::Material);
	desc.buffer_usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
	desc.memory_usage = VMA_MEMORY_USAGE_CPU_TO_GPU;

	m_buffer = std::make_unique<Buffer>(p_device, desc);
}

Buffer &Material::GetBuffer()
{
	return *m_buffer;
}

AlphaMode Material::GetAlphaMode()
{
	return m_alpha_mode;
}

const std::string &Material::GetName() const
{
	return m_name;
}

bool Material::OnImGui(ImGuiContext &context)
{
	ImGui::PushItemWidth(ImGui::GetContentRegionAvailWidth() * 0.6f);

	bool is_update = false;

	const char *const material_types[] = {
	    "Metal-Roughness",
	    "Specular-Glossiness"};
	is_update |= ImGui::Combo("Material Type", reinterpret_cast<int32_t *>(&m_type), material_types, 2);

	int32_t alpha_mode = static_cast<int32_t>(m_alpha_mode) >> 1;

	const char *const alpha_modes[] = {
	    "Opaque",
	    "Masked",
	    "Blend"};
	is_update |= ImGui::Combo("Alpha Mode", &alpha_mode, alpha_modes, 3);
	m_alpha_mode = static_cast<AlphaMode>(1 << alpha_mode);

	if (m_type == MaterialType::MetalRoughnessWorkflow)
	{
		is_update |= ImGui::ColorEdit4("Base Color Factor", glm::value_ptr(m_albedo_factor));
		is_update |= ImGui::DragFloat("Metallic Factor", &m_metallic_factor, 0.001f, 0.f, 1.f, "%.3f");
		is_update |= ImGui::DragFloat("Roughness Factor", &m_roughness_factor, 0.001f, 0.f, 1.f, "%.3f");
	}
	else if (m_type == MaterialType::SpecularGlossinessWorkflow)
	{
		is_update |= ImGui::ColorEdit4("Diffuse Factor", glm::value_ptr(m_albedo_factor));
		is_update |= ImGui::ColorEdit3("Specular Factor", glm::value_ptr(m_specular_factor));
		is_update |= ImGui::DragFloat("Glossiness Factor", &m_glossiness_factor, 0.001f, 0.f, 1.f, "%.3f");
	}

	is_update |= ImGui::ColorEdit3("Emissive Factor", glm::value_ptr(m_emissive_factor));
	is_update |= ImGui::DragFloat("Emissive Strength", &m_emissive_strength, 0.001f, 0.f, 10.f, "%.3f");
	is_update |= ImGui::DragFloat("Alpha Cut Off", &m_alpha_cut_off, 0.001f, 0.f, 1.f, "%.3f");

	if (ImGui::TreeNode("Textures"))
	{
		if (m_type == MaterialType::MetalRoughnessWorkflow)
		{
			if (ImGui::TreeNode("Base Color Texture"))
			{
				is_update |= DrawTextureButton(m_albedo_texture, context, m_manager);
				ImGui::TreePop();
			}
			if (ImGui::TreeNode("Metallic Roughness Texture"))
			{
				is_update |= DrawTextureButton(m_metallic_roughness_texture, context, m_manager);
				ImGui::TreePop();
			}
		}
		else if (m_type == MaterialType::SpecularGlossinessWorkflow)
		{
			if (ImGui::TreeNode("Diffuse Texture"))
			{
				is_update |= DrawTextureButton(m_albedo_texture, context, m_manager);
				ImGui::TreePop();
			}
			if (ImGui::TreeNode("Specular Glossiness Texture"))
			{
				is_update |= DrawTextureButton(m_specular_glossiness_texture, context, m_manager);
				ImGui::TreePop();
			}
		}
		if (ImGui::TreeNode("Normal Texture"))
		{
			is_update |= DrawTextureButton(m_normal_texture, context, m_manager);
			ImGui::TreePop();
		}
		if (ImGui::TreeNode("Emissive Texture"))
		{
			is_update |= DrawTextureButton(m_emissive_texture, context, m_manager);
			ImGui::TreePop();
		}
		ImGui::TreePop();
	}

	ImGui::PopItemWidth();

	if (is_update)
	{
		UpdateBuffer();
	}

	return is_update;
}

void Material::UpdateBuffer()
{
	ShaderInterop::Material material_interop = {};
	material_interop.albedo_factor           = m_albedo_factor;

	material_interop.specular_factor   = m_specular_factor;
	material_interop.glossiness_factor = m_glossiness_factor;

	material_interop.metallic_factor  = m_metallic_factor;
	material_interop.roughness_factor = m_roughness_factor;
	material_interop.type             = static_cast<uint32_t>(m_type);
	material_interop.alpha_mode       = static_cast<uint32_t>(m_alpha_mode);

	material_interop.emissive_factor   = m_emissive_factor;
	material_interop.emissive_strength = m_emissive_strength;

	material_interop.albedo_texture              = m_manager.GetIndex(m_albedo_texture);
	material_interop.normal_texture              = m_manager.GetIndex(m_normal_texture);
	material_interop.emissive_texture            = m_manager.GetIndex(m_emissive_texture);
	material_interop.specular_glossiness_texture = m_manager.GetIndex(m_specular_glossiness_texture);

	material_interop.metallic_roughness_texture = m_manager.GetIndex(m_metallic_roughness_texture);
	material_interop.alpha_cut_off              = m_alpha_cut_off;

	std::memcpy(m_buffer->Map(), &material_interop, sizeof(material_interop));
	m_buffer->Flush(m_buffer->GetSize());
	m_buffer->Unmap();
}
}        // namespace Ilum