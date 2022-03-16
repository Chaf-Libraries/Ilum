#ifndef _GLOBAL_FUNCTION_GLSL_
#define _GLOBAL_FUNCTION_GLSL_

#define PI 3.141592653589793

float RadicalInverse_VdC(uint bits)
{
	bits = (bits << 16u) | (bits >> 16u);
	bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
	bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
	bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
	bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
	return float(bits) * 2.3283064365386963e-10;        // / 0x100000000
}

vec2 Hammersley(uint i, uint N)
{
	return vec2(float(i) / float(N), RadicalInverse_VdC(i));
}

float rand_1to1(float x)
{
	return fract(sin(x) * 43758.5453123);
}

float rand_2to1(vec2 uv)
{
	return fract(sin(dot(uv, vec2(12.9898, 78.233))) * 43758.5453);
}

float rand_3to1(vec3 uvw)
{
	return fract(sin(dot(uvw, vec3(12.9898, 78.233, 144.7272))) * 43758.5453);
}

vec2 poisson_disk_samples_2d(vec2 seed, int samples_num, int rings_num, int step)
{
	float angle  = rand_2to1(seed) * PI * 2.0 + step * PI * 2.0 * float(rings_num) / float(samples_num);
	float radius = (float(step) + 1.0) / float(samples_num);
	return vec2(cos(angle) * sin(angle), cos(angle)) * pow(radius, 0.75);
}

vec3 poisson_disk_samples_3d(vec3 seed, int samples_num, int rings_num, int step)
{
	vec2  angle  = vec2(0.0);
	angle.x      = rand_2to1(seed.xy) * PI * 2.0 + step * PI * 2.0 * float(rings_num) / float(samples_num);
	angle.y      = rand_2to1(seed.yz) * PI * 2.0 + step * PI * 2.0 * float(rings_num) / float(samples_num);
	float radius = (float(step) + 1.0) / float(samples_num);
	return vec3(sin(angle.x) * cos(angle.y), sin(angle.x) * sin(angle.y), cos(angle.x)) * pow(radius, 0.75);
}

vec2 uniform_disk_samples_2d(vec2 seed)
{
	float rand_num = rand_2to1(seed);
	float sample_x = rand_1to1(rand_num);
	float sample_y = rand_1to1(sample_x);

	float angle = sample_x * PI * 2;
	float radius = sqrt(sample_y);

	return vec2(radius * cos(angle), radius * sin(angle));
}

vec3 uniform_disk_samples_3d(vec3 seed)
{
	float rand_num_x = rand_2to1(seed.xy);
	float rand_num_y = rand_2to1(seed.yz);

	float sample_x = rand_2to1(vec2(rand_num_x, rand_num_y));
	float sample_y = rand_1to1(sample_x);
	float sample_z = rand_1to1(sample_y);

	vec2 angle = vec2(sample_x * PI * 2, sample_y * PI * 2);
	float radius = sqrt(sample_z);

	return vec3(sin(angle.x) * cos(angle.y), sin(angle.x) * sin(angle.y), cos(angle.x)) * radius;
}

#endif