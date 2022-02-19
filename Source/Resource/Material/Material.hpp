#pragma once

#include <glm/glm.hpp>

#include <string>

namespace Ilum::Resource
{
// TODO: More powerful material system

struct Material
{
	glm::vec4 base_color     = glm::vec4(1.f);
	glm::vec4 emissive_color = glm::vec4(1.f);

	float emissive_intensity  = 0.f;
	float metallic_factor     = 1.f;
	float roughness_factor    = 1.f;
	float specular_factor     = 0.f;
	float displacement_height = 0.f;

	std::string base_color_map   = "";
	std::string normal_map       = "";
	std::string metallic_map     = "";
	std::string roughness_map    = "";
	std::string emissive_map     = "";
	std::string specular_map     = "";
	std::string ao_map           = "";
	std::string displacement_map = "";
};
}        // namespace Ilum::Graphics