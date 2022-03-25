#ifndef _RANDOM_GLSL
#define _RANDOM_GLSL

// https://dl.acm.org/doi/10.5555/1921479.1921500 
/*GPU Random Numbers via the Tiny Encryption Algorithm*/
uint tea(in uint val0, in uint val1)
{
	uint v0 = val0;
	uint v1 = val1;
	uint s0 = 0;

	for (uint n = 0; n < 16; n++)
	{
		s0 += 0x9e3779b9;
		v0 += ((v1 << 4) + 0xa341316c) ^ (v1 + s0) ^ ((v1 >> 5) + 0xc8013ea4);
		v1 += ((v0 << 4) + 0xad90777d) ^ (v0 + s0) ^ ((v0 >> 5) + 0x7e95761e);
	}

	return v0;
}

uint init_random(in uvec2 resolution, in uvec2 screen_coord, in uint frame)
{
	return tea(screen_coord.y * resolution.x + screen_coord.x, frame);
}

// https://www.pcg-random.org/
uint pcg(inout uint state)
{
	uint prev = state * 747796405u + 2891336453u;
	uint word = ((prev >> ((prev >> 28u) + 4u)) ^ prev) * 277803737u;
	state     = prev;
	return (word >> 22u) ^ word;
}

uvec2 pcg2d(in uvec2 v)
{
	v = v * 1664525u + 1013904223u;
	v.x += v.y * 1664525u;
	v.y += v.x * 1664525u;
	v = v ^ (v >> 16u);
	v.x += v.y * 1664525u;
	v.y += v.x * 1664525u;
	v = v ^ (v >> 16u);
	return v;
}

uvec3 pcg3d(in uvec3 v)
{
	v = v * 1664525u + uvec3(1013904223u);
	v.x += v.y * v.z;
	v.y += v.z * v.x;
	v.z += v.x * v.y;
	v ^= v >> uvec3(16u);
	v.x += v.y * v.z;
	v.y += v.z * v.x;
	v.z += v.x * v.y;
	return v;
}

float rand(inout uint seed)
{
	uint r = pcg(seed);
	return uintBitsToFloat(0x3f800000 | (r >> 9)) - 1.0f;
}

vec2 rand2(inout uint prev)
{
	return vec2(rand(prev), rand(prev));
}

vec3 rand3(inout uint prev)
{
	return vec3(rand(prev), rand(prev), rand(prev));
}

#endif