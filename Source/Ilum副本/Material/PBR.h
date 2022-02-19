#pragma once

#include "Material.h"

#include <glm/glm.hpp>

namespace Ilum::material
{
struct PBRMaterial : public TMaterial<PBRMaterial>
{
	glm::vec4 base_color      = {1.f, 1.f, 1.f, 1.f};
	glm::vec3 emissive_color     = {0.f, 0.f, 0.f};
	float     emissive_intensity  = 0.f;
	float     metallic_factor = 1.f;
	float     roughness_factor = 1.f;
	float     displacement_height = 0.0;

	std::string albedo_map;
	std::string normal_map;
	std::string metallic_map;
	std::string roughness_map;
	std::string emissive_map;
	std::string ao_map;
	std::string displacement_map;
};
}        // namespace Ilum::material