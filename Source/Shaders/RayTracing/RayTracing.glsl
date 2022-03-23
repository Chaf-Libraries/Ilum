#ifndef _RAYTRACING_GLSL_
#	define _RAYTRACING_GLSL_

struct Ray
{
	vec3 origin;
	vec3 direction;
};

struct HitPayLoad
{
	vec3 hitValue;
};

struct PathTracingPayLoad
{
	uint seed;
	float hitT;
	int   primitiveID;
	int   instanceID;
	vec2  bary_coord;
};

struct BxDFSampleRecord
{
	vec3 L;
	vec3 f;
	float pdf;
};

struct LightSampleRecord
{
	vec3 surface_pos;
	vec3 normal;
	vec3 emission;
	float pdf;
};

#endif