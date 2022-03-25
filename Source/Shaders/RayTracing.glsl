#ifndef _RAYTRACING_GLSL_
#define _RAYTRACING_GLSL_

const float ShadowEpsilon = 0.0001;

struct Ray
{
	// r(t) = o + t*d
	vec3  origin;
	vec3  direction;
	float tmin;
	float tmax;
};

struct Intersection
{
	vec3 p;         // Hit point
	vec3 wo;        // Negative ray direction
	vec3 n;         // The surface normal at the point
};

/*RayTracing Gems Chapter 06*/
// Offset current point p along normal n
vec3 OffsetRay(vec3 p, vec3 n)
{
	float origin      = 1.0 / 32.0;
	float float_scale = 1.0 / 65536.0;
	float int_scale   = 256.0;

	ivec3 of_i = ivec3(int_scale * n.x, int_scale * n.y, int_scale * n.z);

	vec3 p_i = vec3(
	    intBitsToFloat(floatToIntBits(p.x) + ((p.x < 0) ? -of_i.x : of_i.x)),
	    intBitsToFloat(floatToIntBits(p.y) + ((p.y < 0) ? -of_i.y : of_i.y)),
	    intBitsToFloat(floatToIntBits(p.z) + ((p.z < 0) ? -of_i.z : of_i.z)));

	return vec3(abs(p.x) < origin ? p.x + float_scale * n.x : p_i.x,
	            abs(p.y) < origin ? p.y + float_scale * n.y : p_i.y,
	            abs(p.z) < origin ? p.z + float_scale * n.z : p_i.z);
}

// Spawn ray along direction d
Ray SpawnRay(in vec3 d, in Intersection intersection)
{
	vec3 o = OffsetRay(intersection.p, intersection.n);

	Ray ray;
	ray.origin    = o;
	ray.direction = d;
	ray.tmin      = 0.0001;
	ray.tmax      = 10000.0;

	return ray;
}

// Spawn ray to point p
Ray SpawnRayTo(in vec3 p, in Intersection intersection)
{
	vec3 origin = OffsetRay(intersection.p, intersection.n);
	vec3 d      = p - origin;

	Ray ray;
	ray.origin    = origin;
	ray.direction = normalize(d);
	ray.tmin      = 0.0001;
	ray.tmax      = length(d);

	return ray;
}

// Spawn ray to another intersection
Ray SpawnRayTo(in Intersection from, in Intersection to)
{
	vec3 origin = OffsetRay(from.p, from.n);
	vec3 target = OffsetRay(to.p, to.n);
	vec3 d      = target - origin;

	Ray ray;
	ray.origin    = origin;
	ray.direction = normalize(d);
	ray.tmin      = 0.0001;
	ray.tmax      = length(d);

	return ray;
}

#endif