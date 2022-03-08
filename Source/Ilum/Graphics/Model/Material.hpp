#pragma once

#include <glm/glm.hpp>

#include <string>

namespace Ilum
{
enum class BxDFType
{
	Disney,
	Lambertian
};

enum TextureType
{
	BaseColor,
	Normal,
	Metallic,
	Roughness,
	Emissive,
	AmbientOcclusion,
	Displacement,
	MaxNum
};

struct Material
{
	BxDFType type = BxDFType::Disney;

	glm::vec4 base_color = {1.f, 1.f, 1.f, 1.f};

	glm::vec3 emissive_color     = {0.f, 0.f, 0.f};
	float     emissive_intensity = 0.f;

	float displacement           = 0.f;
	float subsurface             = 0.f;
	float metallic               = 1.f;
	float specular               = 0.f;
	float specular_tint          = 0.f;
	float roughness              = 1.f;
	float anisotropic            = 0.f;
	float sheen                  = 0.f;
	float sheen_tint             = 0.f;
	float clearcoat              = 0.f;
	float clearcoat_gloss        = 0.f;
	float transmission           = 0.f;
	float transmission_roughness = 0.f;

	std::string textures[TextureType::MaxNum];

	inline static bool update = false;
};

}        // namespace Ilum