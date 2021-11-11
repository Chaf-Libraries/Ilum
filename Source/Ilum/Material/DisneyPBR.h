#pragma once

#include "Material.h"

#include <glm/glm.hpp>

namespace Ilum::material
{
struct DisneyPBR : public TMaterial<DisneyPBR>
{
	glm::vec4 base_color      = {1.f, 1.f, 1.f, 1.f};
	float     metallic_factor = 1.f;
	float     roughness_factor = 1.f;
	float     displacement_height = 0.0;

	std::string albedo_map;
	std::string normal_map;
	std::string metallic_map;
	std::string roughness_map;
	std::string displacement_map;
};
}        // namespace Ilum::material