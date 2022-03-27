#ifndef _GEOMETRY_GLSL
#define _GEOMETRY_GLSL

const float PI = 3.14159265358979323846;
const float InvPI          = 0.31830988618379067154;
const float Inv2PI         = 0.15915494309189533577;
const float Inv4PI         = 0.07957747154594766788;
const float PIOver2        = 1.57079632679489661923;
const float PIOver4        = 0.78539816339744830961;
const float Sqrt2          = 1.41421356237309504880;
const float Infinity       = 1e32;

const float ShadowEpsilon = 0.0001;

struct Ray
{
	vec3 origin;
	vec3 direction;
	float tmin;
	float tmax;
};

struct Vertex
{
	vec4 position;
	vec4 texcoord;
	vec4 normal;
	vec4 tangent;
	vec4 bitangent;
};

float AbsCosTheta(vec3 w)
{
	return abs(w.z);
}

float CosTheta(vec3 w)
{
	return w.z;
}

bool SameHemisphere(vec3 w, vec3 wp)
{
	return w.z * wp.z > 0;
}

#endif