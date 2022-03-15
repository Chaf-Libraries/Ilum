#pragma once

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
	glm::vec4 split_depth           = {};
	glm::mat4 view_projection[4]    = {glm::mat4(1.f)};
	glm::vec3 color                 = {1.f, 1.f, 1.f};
	float     intensity             = 1.f;
	alignas(16) glm::vec3 direction = {1.f, 1.f, 1.f};
	alignas(16) glm::vec3 position  = {0.f, 0.f, 0.f};
};

struct PointLight : public Light
{
	glm::vec3 color          = {1.f, 1.f, 1.f};
	float     intensity      = 1.f;
	glm::vec3 position       = {0.f, 0.f, 0.f};
	float     constant       = 1.0f;
	alignas(16) float linear = 0.09f;
	float quadratic          = 0.032f;
};

struct SpotLight : public Light
{
	glm::mat4 view_projection = glm::mat4(1.f);
	glm::vec3 color           = {1.f, 1.f, 1.f};
	float     intensity       = 1.f;
	glm::vec3 position        = {0.f, 0.f, 0.f};
	float     cut_off         = glm::cos(glm::radians(12.5f));
	glm::vec3 direction       = {1.f, 1.f, 1.f};
	float     outer_cut_off   = glm::cos(glm::radians(17.5f));
};
}        // namespace Ilum::cmpt