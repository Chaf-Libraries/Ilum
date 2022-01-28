#pragma once

#include <string>

namespace Ilum::Asset
{
enum class ShadingMode
{
	None,
	Flat,
	Gouraud,
	Phong,
	Blinn_Phong,
	Toon,
	OrenNayar,
	Minnaert,
	CookTorrance,
	Fresnel,
	PBR_BRDF
};

struct Material
{
	glm::vec4 base_color     = glm::vec4(1.f);
	glm::vec3 emissive_color = glm::vec4(1.f);

	float metallic_factor     = 1.f;
	float roughness_factor    = 1.f;
	float emissive_intensity  = 0.f;
	float displacement_height = 0.f;
	float specular_factor     = 0.f;
	float glossiness_factor   = 0.f;

	// Texture
	// Diffuse: light equation diffuse item or PBR Specular/Glossiness
	std::string diffuse_map_uri = "";
	// Specular: light equation specular item or PBR Specular/Glossiness
	std::string specular_map_uri = "";
	// Ambient: combined with the result of the ambient lighting equation
	std::string ambient_map_uri = "";
	// Emissive: added ti the result of the lighting calculation
	std::string emissive_map_uri = "";
	// Height: higher gray-scale values stand for higher elevations from the base height.
	std::string height_map_uri = "";
	// Normal: tangent space normal
	std::string normal_map_uri = "";
	// Shinniess: the exponent of the specular (phong) lighting equation.
	std::string shininess_map_uri = "";
	// Opacity: defines per-pixel opacity, 'white' means opaque and 'black' means 'transparency'
	std::string opacity_map_uri = "";
	// Displacement: Higher color values stand for higher vertex displacements.
	std::string displacement_map_uri = "";
	// Ambient Occlusion: contains a scaling value for the final color value of a pixel. Its intensity is not affected by incoming light.
	std::string ao_map_uri = "";

	std::string base_color_map_uri = "";
	std::string metallic_map_uri   = "";
	std::string roughness_map_uri  = "";

	ShadingMode shading_mode = ShadingMode::None;

	bool two_sided = false;
	bool wireframe = false;
	bool opacity   = true;
};
}        // namespace Ilum::Asset