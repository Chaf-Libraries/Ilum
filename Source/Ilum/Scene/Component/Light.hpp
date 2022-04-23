#pragma once

#include "ComponentSerializer.hpp"

#include "Graphics/Image/Image.hpp"

#include <glm/glm.hpp>

namespace Ilum::cmpt
{
struct Light
{
	inline static std::atomic<bool> update = false;
};

struct DirectionalLight : public Light
{
	glm::vec4 split_depth = {};

	glm::mat4 view_projection[4] = {glm::mat4(1.f)};
	glm::vec4 shadow_cam_pos[4]  = {glm::vec4(1.f)};

	glm::vec3 color     = {1.f, 1.f, 1.f};
	float     intensity = 1.f;

	glm::vec3 direction   = {1.f, 1.f, 1.f};
	int32_t   shadow_mode = 2;        // 0 - no shadow, 1 - hard shadow, 2 - PCF, 3 - PCSS

	float   filter_scale  = 3.f;
	int32_t filter_sample = 20;
	int32_t sample_method = 1;        // 0 - Uniform, 1 - Poisson Disk
	float   light_size    = 10.f;

	alignas(16) glm::vec3 position = {0.f, 0.f, 0.f};

	template <class Archive>
	void serialize(Archive &ar)
	{
		glm::serialize(ar, color, direction, position);
		ar(intensity, shadow_mode, filter_scale, filter_sample, sample_method, light_size);
	}
};

struct PointLight : public Light
{
	glm::vec3 color     = {1.f, 1.f, 1.f};
	float     intensity = 1.f;

	glm::vec3 position = {0.f, 0.f, 0.f};
	float     constant = 1.0f;

	float   linear       = 0.09f;
	float   quadratic    = 0.032f;
	int32_t shadow_mode  = 2;        // 0 - no shadow, 1 - hard shadow, 2 - PCF, 3 - PCSS
	float   filter_scale = 3.f;

	alignas(16) int32_t filter_sample = 20;
	int32_t sample_method             = 1;        // 0 - Uniform, 1 - Poisson Disk
	float   light_size                = 10.f;

	template <class Archive>
	void serialize(Archive &ar)
	{
		glm::serialize(ar, color, position);
		ar(intensity, constant, linear, quadratic, shadow_mode, filter_scale, filter_sample, sample_method, light_size);
	}
};

struct SpotLight : public Light
{
	glm::mat4 view_projection = glm::mat4(1.f);

	glm::vec3 color     = {1.f, 1.f, 1.f};
	float     intensity = 1.f;

	glm::vec3 position = {0.f, 0.f, 0.f};
	float     cut_off  = glm::cos(glm::radians(12.5f));

	glm::vec3 direction     = {1.f, 1.f, 1.f};
	float     outer_cut_off = glm::cos(glm::radians(17.5f));

	int32_t shadow_mode   = 2;        // 0 - no shadow, 1 - hard shadow, 2 - PCF, 3 - PCSS
	float   filter_scale  = 3.f;
	int32_t filter_sample = 20;
	int32_t sample_method = 1;        // 0 - Uniform, 1 - Poisson Disk

	alignas(16) float light_size = 10.f;

	template <class Archive>
	void serialize(Archive &ar)
	{
		glm::serialize(ar, color, position, direction);
		ar(intensity, cut_off, outer_cut_off, shadow_mode, filter_scale, filter_sample, sample_method, light_size);
	}
};

struct AreaLight : public Light
{
	glm::vec3 color     = {1.f, 1.f, 1.f};
	float     intensity = 1.f;
	glm::vec4 corners[4];

	alignas(16) enum class AreaLightType : uint32_t {
		Rectangle,
		Ellipse
	} shape             = AreaLightType::Rectangle;

	uint32_t texture_id;

	template <class Archive>
	void serialize(Archive &ar)
	{
		glm::serialize(ar, color, corners[0], corners[1], corners[2], corners[3]);
		ar(intensity, shape);
	}
};
}        // namespace Ilum::cmpt