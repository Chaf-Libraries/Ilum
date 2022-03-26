#ifndef _LIGHTS_GLSL
#define _LIGHTS_GLSL

struct DirectionalLight
{
	vec4  split_depth;
	mat4  view_projection[4];
	vec3  color;
	float intensity;
	vec3  direction;

	////
	// Rasterization Shadow
	int   shadow_mode;        // 0 - no shadow, 1 - hard shadow, 2 - PCF, 3 - PCSS
	float filter_scale;
	int   filter_sample;
	int   sample_method;        // 0 - Uniform, 1 - Poisson Disk
	float light_size;
	////

	vec3  position;
};

struct PointLight
{
	vec3  color;
	float intensity;
	vec3  position;
	float constant;
	float linear_;
	float quadratic;

	// Rasterization Shadow
	int   shadow_mode;        // 0 - no shadow, 1 - hard shadow, 2 - PCF, 3 - PCSS
	float filter_scale;
	int   filter_sample;
	int   sample_method;        // 0 - Uniform, 1 - Poisson Disk
	float light_size;
};

struct SpotLight
{
	mat4  view_projection;
	vec3  color;
	float intensity;
	vec3  position;
	float cut_off;
	vec3  direction;
	float outer_cut_off;

	// Rasterization Shadow
	int   shadow_mode;        // 0 - no shadow, 1 - hard shadow, 2 - PCF, 3 - PCSS
	float filter_scale;
	int   filter_sample;
	int   sample_method;        // 0 - Uniform, 1 - Poisson Disk
	float light_size;
};

#endif