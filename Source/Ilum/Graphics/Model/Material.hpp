#pragma once

#include <glm/glm.hpp>

#include <string>

#include <cereal/cereal.hpp>

namespace Ilum
{
enum class BxDFType : uint32_t
{
	Matte,
	Plastic,
	Metal,
	Mirror,
	Substrate,
	Glass,
	Disney
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

	glm::vec3 data = {0.f, 0.f, 0.f};

	glm::vec4 base_color = {1.f, 1.f, 1.f, 1.f};

	glm::vec3 emissive_color     = {0.f, 0.f, 0.f};
	float     emissive_intensity = 1.f;

	float displacement           = 0.f;

	float subsurface             = 0.f;
	float anisotropic = 0.f;

	float metallic               = 1.f;
	float roughness              = 1.f;

	float specular               = 0.f;
	float specular_tint          = 0.f;

	float sheen                  = 0.f;
	float sheen_tint             = 0.f;

	float clearcoat              = 0.f;
	float clearcoat_gloss        = 0.f;

	float specular_transmission           = 0.f;
	float diffuse_transmission = 0.f;

	float flatness = 0.0;
	float thin     = 0.0;
	float refraction = 1.5;

	std::string textures[TextureType::MaxNum];

	inline static bool update = false;

	template<class Archive>
	void serialize(Archive& ar)
	{
		ar(type,
		   emissive_intensity,
		   displacement,
		   subsurface,
		   metallic,
		   specular,
		   specular_tint,
		   roughness,
		   anisotropic,
		   sheen,
		   sheen_tint,
		   clearcoat,
		   clearcoat_gloss,
		   specular_transmission,
		   diffuse_transmission,
		   flatness,
		   thin,
		   refraction,
		   data.x, data.y, data.z,
		   base_color.x, base_color.y, base_color.z, base_color.w,
		   emissive_color.x, emissive_color.y, emissive_color.z,
		   textures);
	}
};

}        // namespace Ilum
