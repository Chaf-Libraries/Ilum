#version 460
#extension GL_EXT_ray_tracing : enable
#extension GL_EXT_nonuniform_qualifier : enable

#include "../RayTracing.glsl"

layout(location = 1) rayPayloadInEXT ShadowPayload prd;

void main()
{
    prd.visibility = true;
}