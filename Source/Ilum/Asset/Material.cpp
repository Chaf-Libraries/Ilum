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
	is_update |= ImGui::DragFloat("IOR", &m_ior, 0.001f, 0.f, 10.f, "%.3f");

	if (m_type == MaterialType::MetalRoughnessWorkflow)
	{
		if (ImGui::TreeNode("PBR - Metal Roughness Workflow"))
		{
			is_update |= ImGui::ColorEdit4("Base Color Factor", glm::value_ptr(m_pbr_base_color_factor));
			is_update |= ImGui::DragFloat("Metallic Factor", &m_pbr_metallic_factor, 0.001f, 0.f, 1.f, "%.3f");
			is_update |= ImGui::DragFloat("Roughness Factor", &m_pbr_roughness_factor, 0.001f, 0.f, 1.f, "%.3f");
			if (ImGui::TreeNode("Base Color Texture"))
			{
				is_update |= DrawTextureButton(m_pbr_base_color_texture, context, m_manager);
				ImGui::TreePop();
			}
			if (ImGui::TreeNode("Metallic Roughness Texture"))
			{
				is_update |= DrawTextureButton(m_pbr_metallic_roughness_texture, context, m_manager);
				ImGui::TreePop();
			}
			if (ImGui::TreeNode("Normal Texture"))
			{
				is_update |= DrawTextureButton(m_normal_texture, context, m_manager);
				ImGui::TreePop();
			}
			if (ImGui::TreeNode("Occlusion Texture"))
			{
				is_update |= DrawTextureButton(m_occlusion_texture, context, m_manager);
				ImGui::TreePop();
			}
			ImGui::TreePop();
		}
	}
	else if (m_type == MaterialType::SpecularGlossinessWorkflow)
	{
		if (ImGui::TreeNode("PBR - Specular Glossiness Workflow"))
		{
			is_update |= ImGui::ColorEdit4("Diffuse Factor", glm::value_ptr(m_pbr_diffuse_factor));
			is_update |= ImGui::ColorEdit3("Specular Factor", glm::value_ptr(m_pbr_specular_factor));
			is_update |= ImGui::DragFloat("Glossiness Factor", &m_pbr_glossiness_factor, 0.001f, 0.f, 1.f, "%.3f");
			if (ImGui::TreeNode("Diffuse Texture"))
			{
				is_update |= DrawTextureButton(m_pbr_diffuse_texture, context, m_manager);
				ImGui::TreePop();
			}
			if (ImGui::TreeNode("Specular Glossiness Texture"))
			{
				is_update |= DrawTextureButton(m_pbr_specular_glossiness_texture, context, m_manager);
				ImGui::TreePop();
			}
			if (ImGui::TreeNode("Normal Texture"))
			{
				is_update |= DrawTextureButton(m_normal_texture, context, m_manager);
				ImGui::TreePop();
			}
			if (ImGui::TreeNode("Occlusion Texture"))
			{
				is_update |= DrawTextureButton(m_occlusion_texture, context, m_manager);
				ImGui::TreePop();
			}
			ImGui::TreePop();
		}
	}

	if (ImGui::TreeNode("Emissive"))
	{
		is_update |= ImGui::ColorEdit3("Factor", glm::value_ptr(m_emissive_factor));
		is_update |= ImGui::DragFloat("Strength", &m_emissive_strength, 0.001f, 0.f, 10.f, "%.3f");
		is_update |= ImGui::DragFloat("Alpha Cut Off", &m_alpha_cut_off, 0.001f, 0.f, 1.f, "%.3f");
		if (ImGui::TreeNode("Texture"))
		{
			is_update |= DrawTextureButton(m_emissive_texture, context, m_manager);
			ImGui::TreePop();
		}
		ImGui::TreePop();
	}

	if (ImGui::TreeNode("Sheen"))
	{
		is_update |= ImGui::ColorEdit3("Color Factor", glm::value_ptr(m_sheen_color_factor));
		is_update |= ImGui::DragFloat("Roughness Factor", &m_sheen_roughness_factor, 0.001f, 0.f, 1.f, "%.3f");
		if (ImGui::TreeNode("Texture"))
		{
			is_update |= DrawTextureButton(m_sheen_texture, context, m_manager);
			ImGui::TreePop();
		}
		if (ImGui::TreeNode("Roughness Texture"))
		{
			is_update |= DrawTextureButton(m_sheen_roughness_texture, context, m_manager);
			ImGui::TreePop();
		}
		ImGui::TreePop();
	}

	if (ImGui::TreeNode("Clearcoat"))
	{
		is_update |= ImGui::DragFloat("Factor", &m_clearcoat_factor, 0.001f, 0.f, 1.f, "%.3f");
		is_update |= ImGui::DragFloat("Roughness Factor", &m_clearcoat_roughness_factor, 0.001f, 0.f, 1.f, "%.3f");
		if (ImGui::TreeNode("Texture"))
		{
			is_update |= DrawTextureButton(m_clearcoat_texture, context, m_manager);
			ImGui::TreePop();
		}
		if (ImGui::TreeNode("Roughness Texture"))
		{
			is_update |= DrawTextureButton(m_clearcoat_roughness_texture, context, m_manager);
			ImGui::TreePop();
		}
		if (ImGui::TreeNode("Normal Texture"))
		{
			is_update |= DrawTextureButton(m_clearcoat_normal_texture, context, m_manager);
			ImGui::TreePop();
		}
		ImGui::TreePop();
	}

	if (ImGui::TreeNode("Specular"))
	{
		is_update |= ImGui::ColorEdit3("Color Factor", glm::value_ptr(m_specular_color_factor));
		is_update |= ImGui::DragFloat("Factor", &m_specular_factor, 0.001f, 0.f, 1.f, "%.3f");
		if (ImGui::TreeNode("Texture"))
		{
			is_update |= DrawTextureButton(m_specular_texture, context, m_manager);
			ImGui::TreePop();
		}
		if (ImGui::TreeNode("Color Texture"))
		{
			is_update |= DrawTextureButton(m_specular_color_texture, context, m_manager);
			ImGui::TreePop();
		}
		ImGui::TreePop();
	}

	if (ImGui::TreeNode("Transmission"))
	{
		is_update |= ImGui::DragFloat("Factor", &m_transmission_factor, 0.001f, 0.f, 1.f, "%.3f");
		if (ImGui::TreeNode("Texture"))
		{
			is_update |= DrawTextureButton(m_transmission_texture, context, m_manager);
			ImGui::TreePop();
		}
		ImGui::TreePop();
	}

	if (ImGui::TreeNode("Volume"))
	{
		is_update |= ImGui::ColorEdit3("Attenuation Color", glm::value_ptr(m_attenuation_color));
		is_update |= ImGui::DragFloat("Thickness Factor", &m_thickness_factor, 0.001f, 0.f, 1.f, "%.3f");
		is_update |= ImGui::DragFloat("Attenuation Distance", &m_attenuation_distance, 0.001f, 0.f, 1.f, "%.3f");
		ImGui::TreePop();
	}

	if (ImGui::TreeNode("Iridescence"))
	{
		is_update |= ImGui::DragFloat("Factor", &m_iridescence_factor, 0.001f, 0.f, 1.f, "%.3f");
		is_update |= ImGui::DragFloat("IOR", &m_iridescence_ior, 0.001f, 0.f, 1.f, "%.3f");
		is_update |= ImGui::DragFloat("Thickness Min", &m_iridescence_thickness_min, 0.001f, 0.f, 1.f, "%.3f");
		is_update |= ImGui::DragFloat("Thickness Max", &m_iridescence_thickness_max, 0.001f, 0.f, 1.f, "%.3f");
		if (ImGui::TreeNode("Thickness Texture"))
		{
			is_update |= DrawTextureButton(m_iridescence_thickness_texture, context, m_manager);
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

	material_interop.type                            = static_cast<uint32_t>(m_type);
	material_interop.pbr_diffuse_factor              = m_pbr_diffuse_factor;
	material_interop.pbr_specular_factor             = m_pbr_specular_factor;
	material_interop.pbr_glossiness_factor           = m_pbr_glossiness_factor;
	material_interop.pbr_diffuse_texture             = m_manager.GetIndex(m_pbr_diffuse_texture);
	material_interop.pbr_specular_glossiness_texture = m_manager.GetIndex(m_pbr_specular_glossiness_texture);

	material_interop.pbr_base_color_factor          = m_pbr_base_color_factor;
	material_interop.pbr_metallic_factor            = m_pbr_metallic_factor;
	material_interop.pbr_roughness_factor           = m_pbr_roughness_factor;
	material_interop.pbr_base_color_texture         = m_manager.GetIndex(m_pbr_base_color_texture);
	material_interop.pbr_metallic_roughness_texture = m_manager.GetIndex(m_pbr_metallic_roughness_texture);

	material_interop.emissive_factor   = m_emissive_factor;
	material_interop.emissive_strength = m_emissive_strength;
	material_interop.emissive_texture  = m_manager.GetIndex(m_emissive_texture);

	material_interop.sheen_color_factor      = m_sheen_color_factor;
	material_interop.sheen_roughness_factor  = m_sheen_roughness_factor;
	material_interop.sheen_texture           = m_manager.GetIndex(m_sheen_texture);
	material_interop.sheen_roughness_texture = m_manager.GetIndex(m_sheen_roughness_texture);

	material_interop.clearcoat_factor            = m_clearcoat_factor;
	material_interop.clearcoat_roughness_factor  = m_clearcoat_roughness_factor;
	material_interop.clearcoat_texture           = m_manager.GetIndex(m_clearcoat_texture);
	material_interop.clearcoat_roughness_texture = m_manager.GetIndex(m_clearcoat_roughness_texture);
	material_interop.clearcoat_normal_texture    = m_manager.GetIndex(m_clearcoat_normal_texture);

	material_interop.specular_factor        = m_specular_factor;
	material_interop.specular_color_factor  = m_specular_color_factor;
	material_interop.specular_texture       = m_manager.GetIndex(m_specular_texture);
	material_interop.specular_color_texture = m_manager.GetIndex(m_specular_color_texture);

	material_interop.transmission_factor  = m_transmission_factor;
	material_interop.transmission_texture = m_manager.GetIndex(m_transmission_texture);

	material_interop.thickness_factor  = m_thickness_factor;
	material_interop.attenuation_color = m_attenuation_color;
	material_interop.attenuation_distance = m_attenuation_distance;

	material_interop.iridescence_factor = m_iridescence_factor;
	material_interop.iridescence_ior    = m_iridescence_ior;
	material_interop.iridescence_thickness_min = m_iridescence_thickness_min;
	material_interop.iridescence_thickness_max = m_iridescence_thickness_max;
	material_interop.iridescence_thickness_texture = m_manager.GetIndex(m_iridescence_thickness_texture);

	material_interop.ior = m_ior;
	material_interop.alpha_cut_off = m_alpha_cut_off;
	material_interop.alpha_mode  = static_cast<uint32_t>(m_alpha_mode);

	material_interop.normal_texture    = m_manager.GetIndex(m_normal_texture);
	material_interop.occlusion_texture           = m_manager.GetIndex(m_occlusion_texture);

	material_interop.unlit = static_cast<uint32_t>(m_unlit);
	material_interop.thin  = static_cast<uint32_t>(m_thin);

	std::memcpy(m_buffer->Map(), &material_interop, sizeof(material_interop));
	m_buffer->Flush(m_buffer->GetSize());
	m_buffer->Unmap();
}
}        // namespace Ilum