#pragma once

#include "Material.h"

#include <glm/glm.hpp>

#include <string>

namespace Ilum::material
{
struct BlinnPhong : public TMaterial<BlinnPhong>
{
	// Factor
	struct Data
	{
		glm::vec4 base_color = {1.f, 1.f, 1.f, 1.f};

		glm::vec3 emissive           = {0.f, 0.f, 0.f};
		float     emissive_intensity = 0.f;

		glm::vec3 specular           = {1.f, 1.f, 1.f};
		float     displacement_scale = 0.f;

		// Texture
		uint32_t diffuse_map      = std::numeric_limits<uint32_t>::max();
		uint32_t normal_map       = std::numeric_limits<uint32_t>::max();
		uint32_t specular_map     = std::numeric_limits<uint32_t>::max();
		uint32_t displacement_map = std::numeric_limits<uint32_t>::max();

		float shininess    = 32.f;
		float reflectivity = 1.f;
		bool  normal_type  = false;
		bool  flat_shading = false;
	} material_data;

	virtual size_t size() override
	{
		return sizeof(material_data);
	}

	virtual void *data() override
	{
		return &material_data;
	}

	std::string diffuse_map_path      = "";
	std::string normal_map_path       = "";
	std::string specular_map_path     = "";
	std::string displacement_map_path = "";
};
}        // namespace Ilum::material