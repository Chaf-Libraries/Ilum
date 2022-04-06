#include "../Common.hlsli"

[shader("closesthit")]
void main(inout RayPayload rayPayload : SV_RayPayload, BuiltInTriangleIntersectionAttributes Attributes)
{
    rayPayload.hitT = RayTCurrent();
    rayPayload.primitiveID = PrimitiveIndex();
    rayPayload.instanceID = InstanceIndex();
    rayPayload.baryCoord = Attributes.barycentrics;
    rayPayload.objectToWorld = ObjectToWorld4x3();
    rayPayload.worldToObject = WorldToObject4x3();
}
