#ifndef _INTERACTION_GLSL_
#define _INTERACTION_GLSL_

#include "Geometry.glsl"
#include "GlobalBuffer.glsl"

////////////////// Interaction //////////////////
struct Interaction
{
	vec3 normal;	// normal from vertex attribute
	vec3 geom_normal;	// normal from vertex cross product
	vec3 position;
	vec2 texcoord;
	vec3 tangent;
	vec3 bitangent;
	MaterialData material;
};

/*RayTracing Gems Chapter 06*/
// Offset current point p along normal n



#endif