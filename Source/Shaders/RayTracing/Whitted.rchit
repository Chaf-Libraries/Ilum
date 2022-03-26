#version 460
#extension GL_EXT_ray_tracing : enable
#extension GL_GOOGLE_include_directive : enable

#include "../RayTracing.glsl"

layout(location = 0) rayPayloadInEXT RayPayload ray_payload;

hitAttributeEXT vec2 bary;

void main()
{
	ray_payload.hitT = gl_HitTEXT;
	ray_payload.primitiveID = gl_PrimitiveID;
	ray_payload.instanceID = gl_InstanceID;
	ray_payload.baryCoord = bary;
	ray_payload.objectToWorld = gl_ObjectToWorldEXT;
	ray_payload.worldToObject = gl_WorldToObjectEXT;
}
