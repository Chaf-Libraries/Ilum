#pragma once

#include <RHI/Buffer.hpp>
#include <RHI/ImGuiContext.hpp>

#include <glm/glm.hpp>

namespace Ilum
{
class AssetManager;

enum class AlphaMode : uint32_t
{
	Opaque,
	Masked,
	Blend
};

inline AlphaMode operator|(const AlphaMode &lhs, const AlphaMode &rhs)
{
	return static_cast<AlphaMode>(static_cast<uint32_t>(lhs) | static_cast<uint32_t>(rhs));
}

inline AlphaMode operator&(const AlphaMode &lhs, const AlphaMode &rhs)
{
	return static_cast<AlphaMode>(static_cast<uint32_t>(lhs) & static_cast<uint32_t>(rhs));
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

	const std::string &GetName() const;

	bool OnImGui(ImGuiContext &context);

	void UpdateBuffer();

	template <class Archive>
	void serialize(Archive ar)
	{
	}

  private:
	AssetManager &m_manager;
	RHIDevice    *p_device = nullptr;

  private:
	MaterialType m_type = MaterialType::MetalRoughnessWorkflow;

	std::string m_name;

	glm::vec4 m_albedo_factor = glm::vec4(1.f);        // Albedo & Diffuse

	// PBR Specular Glossiness
	glm::vec3 m_specular_factor   = glm::vec3(0.f);
	float     m_glossiness_factor = 0.f;

	// PBR Metallic Roughness
	float m_metallic_factor  = 0.f;
	float m_roughness_factor = 1.f;

	glm::vec3 m_emissive_factor   = glm::vec3(0.f);
	float     m_emissive_strength = 0.f;

	float m_alpha_cut_off = 0.5f;

	Texture  *m_albedo_texture              = nullptr;
	Texture  *m_normal_texture              = nullptr;
	Texture  *m_emissive_texture            = nullptr;
	Texture  *m_specular_glossiness_texture = nullptr;
	Texture  *m_metallic_roughness_texture  = nullptr;
	AlphaMode m_alpha_mode                  = AlphaMode::Opaque;

	std::unique_ptr<Buffer> m_buffer = nullptr;
};
}        // namespace Ilum