#pragma once

#include <RHI/AccelerateStructure.hpp>
#include <RHI/Buffer.hpp>
#include <RHI/Texture.hpp>

#include <glm/glm.hpp>

namespace Ilum
{
struct Vertex
{
	alignas(16) glm::vec3 position;
	alignas(16) glm::vec2 texcoord;
	alignas(16) glm::vec3 normal;
	alignas(16) glm::vec3 tangent;
};

struct Material
{
	enum class AlphaMode
	{
		Opaque,
		Masked,
		Blend
	};

	enum class MaterialType
	{
		Disney
	};

	glm::vec4 base_color      = glm::vec4(1.f);
	glm::vec4 emissive_factor = glm::vec4(0.f, 0.f, 0.f, 1.f);

	MaterialType type          = MaterialType::Disney;
	float        metalness     = 0.f;
	float        roughness     = 1.f;
	float        alpha_cut_off = 0.5f;

	Texture  *albedo              = nullptr;
	Texture  *normal              = nullptr;
	Texture  *roughness_metalness = nullptr;
	AlphaMode alpha_mode          = AlphaMode::Opaque;
};

struct SubMesh
{
	int32_t material_id = -1;

	uint32_t vertex_offset = 0;
	uint32_t vertex_count  = 0;

	uint32_t index_offset = 0;
	uint32_t index_count  = 0;

	uint32_t mehslet_offset = 0;
	uint32_t meshlet_count  = 0;

	std::unique_ptr<AccelerationStructure> bottom_level_acceleration_structure = nullptr;
};

class Mesh
{
  public:
	Mesh(RHIDevice *device, const std::string &filename);
	~Mesh() = default;

	SubMesh  &GetSubMesh(uint32_t index);
	Material &GetMaterial(uint32_t index);

	const std::string &GetName() const;

  private:
	RHIDevice *p_device = nullptr;

	std::string m_name;

	std::unique_ptr<Buffer> m_vertex_buffer;
	std::vector<SubMesh>    m_submeshes;
	std::vector<Material*>   m_materials;
};
}        // namespace Ilum