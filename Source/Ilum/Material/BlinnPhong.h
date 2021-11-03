#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <rttr/registration.h>

namespace Ilum::material
{
struct BlinnPhong
{
	// Factor
	float base_color[4] = {1.f, 1.f, 1.f, 1.f};

	float emissive[3]        = {0.f, 0.f, 0.f};
	float emissive_intensity = 0.f;

	float specular[3]        = {1.f, 1.f, 1.f};
	float displacement_scale = 0.f;

	// Texture
	uint32_t diffuse_map      = std::numeric_limits<uint32_t>::infinity();
	uint32_t normal_map       = std::numeric_limits<uint32_t>::infinity();
	uint32_t specular_map     = std::numeric_limits<uint32_t>::infinity();
	uint32_t displacement_map = std::numeric_limits<uint32_t>::infinity();

	float shininess = 32.f;
	int32_t normal_type = 0;
	float   normal_scale[2] = {1.f, 1.f};
};

RTTR_REGISTRATION
{
	rttr::registration::class_<BlinnPhong>("BlinnPhong")
	    .constructor<>()
	    .property("base_color", &BlinnPhong::base_color)
	    .property("diffuse_map", &BlinnPhong::diffuse_map)
	    .property("emissive", &BlinnPhong::emissive)
	    .property("emissive_intensity", &BlinnPhong::emissive_intensity)
	    .property("specular", &BlinnPhong::specular)
	    .property("specular_map", &BlinnPhong::specular_map)
	    .property("normal_map", &BlinnPhong::normal_map)
	    .property("normal_type", &BlinnPhong::normal_type)
	    .property("normal_scale", &BlinnPhong::normal_scale)
	    .property("displacement_map", &BlinnPhong::displacement_map)
	    .property("displacement_scale", &BlinnPhong::displacement_scale)
	    .property("shininess", &BlinnPhong::shininess);
}
}        // namespace Ilum::material