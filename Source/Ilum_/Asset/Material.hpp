#pragma once

#include <RHI/Buffer.hpp>
#include <RHI/ImGuiContext.hpp>

#include <glm/glm.hpp>

namespace Ilum
{
class AssetManager;

enum class AlphaMode : uint32_t
{
	Opaque = 1,
	Masked = 1 << 1,
	Blend  = 1 << 2
};

inline AlphaMode operator|(const AlphaMode &lhs, const AlphaMode &rhs)
{
	return static_cast<AlphaMode>(static_cast<uint32_t>(lhs) | static_cast<uint32_t>(rhs));
}

inline bool operator&(const AlphaMode &lhs, const AlphaMode &rhs)
{
	return static_cast<uint32_t>(lhs) & static_cast<uint32_t>(rhs);
}

enum class MaterialType
{
	MetalRoughnessWorkflow,
	SpecularGlossinessWorkflow
};

class Material
{
	friend class Scene;

  public:
	Material(RHIDevice *device, AssetManager &manager);
	~Material() = default;

	Buffer &GetBuffer();

	AlphaMode GetAlphaMode();

	const std::string &GetName() const;

	bool OnImGui(ImGuiContext &context);

	void UpdateBuffer();

	template <class Archive>
	void serialize(Archive ar)
	{
		/* ar(m_type, m_name, m_albedo_factor, m_specular_factor,
		   m_glossiness_factor, m_metallic_factor, m_roughness_factor,
		   m_emissive_factor, m_emissive_strength, m_alpha_cut_off,
		   m_alpha_mode);*/
	}

  private:
	AssetManager &m_manager;
	RHIDevice    *p_device = nullptr;

  private:
	MaterialType m_type = MaterialType::MetalRoughnessWorkflow;

	std::string m_name;

	// PBR Specular Glossiness
	glm::vec4 m_pbr_diffuse_factor              = glm::vec4(1.f);
	glm::vec3 m_pbr_specular_factor             = glm::vec3(0.f);
	float     m_pbr_glossiness_factor           = 0.f;
	Texture  *m_pbr_diffuse_texture             = nullptr;
	Texture  *m_pbr_specular_glossiness_texture = nullptr;

	// PBR Metallic Roughness
	glm::vec4 m_pbr_base_color_factor          = glm::vec4(1.f);
	float     m_pbr_metallic_factor            = 0.f;
	float     m_pbr_roughness_factor           = 1.f;
	Texture  *m_pbr_base_color_texture         = nullptr;
	Texture  *m_pbr_metallic_roughness_texture = nullptr;

	// Emissive
	glm::vec3 m_emissive_factor   = glm::vec3(0.f);
	float     m_emissive_strength = 0.f;
	Texture  *m_emissive_texture  = nullptr;

	// Sheen
	glm::vec3 m_sheen_color_factor      = glm::vec3(0.f);
	float     m_sheen_roughness_factor  = 0.f;
	Texture  *m_sheen_texture           = nullptr;
	Texture  *m_sheen_roughness_texture = nullptr;

	// Clear Coat
	float    m_clearcoat_factor            = 0.f;
	float    m_clearcoat_roughness_factor  = 0.f;
	Texture *m_clearcoat_texture           = nullptr;
	Texture *m_clearcoat_roughness_texture = nullptr;
	Texture *m_clearcoat_normal_texture    = nullptr;

	// Specular
	float     m_specular_factor        = 0.f;
	glm::vec3 m_specular_color_factor  = glm::vec3(0.f);
	Texture  *m_specular_texture       = nullptr;
	Texture  *m_specular_color_texture = nullptr;

	// Transmission
	float    m_transmission_factor  = 0.f;
	Texture *m_transmission_texture = nullptr;

	// Volume
	Texture  *m_thickness_texture    = nullptr;
	float     m_thickness_factor     = 0.f;
	glm::vec3 m_attenuation_color    = glm::vec3(0.f);
	float     m_attenuation_distance = 0.f;

	// Iridescence
	float    m_iridescence_factor            = 0.f;
	float    m_iridescence_ior               = 0.f;
	float    m_iridescence_thickness_min     = 0.f;
	float    m_iridescence_thickness_max     = 0.f;
	Texture *m_iridescence_thickness_texture = nullptr;
	Texture *m_iridescence_texture           = nullptr;

	// IOR
	float m_ior = 1.5f;

	// Alpha Test
	float     m_alpha_cut_off = 0.5f;
	AlphaMode m_alpha_mode    = AlphaMode::Opaque;

	bool m_unlit        = false;
	bool m_double_sided = false;

	Texture *m_normal_texture    = nullptr;
	Texture *m_occlusion_texture = nullptr;

	std::unique_ptr<Buffer> m_buffer = nullptr;
};
}        // namespace Ilum