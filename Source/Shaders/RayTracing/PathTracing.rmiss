#version 460
#extension GL_EXT_ray_tracing : enable

#include "RayTracing.glsl"

layout(location = 0) rayPayloadInEXT HitPayLoad prd;

layout(set = 0, binding = 11) uniform samplerCube Skybox;

void main()
{
    prd.hitValue = texture(Skybox, gl_WorldRayDirectionEXT).rgb;
}