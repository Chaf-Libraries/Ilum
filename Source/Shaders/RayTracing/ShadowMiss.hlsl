#include "../Common.hlsli"

[shader("miss")]
void main(inout ShadowPayload shadowPayload : SV_RayPayload)
{
    shadowPayload.visibility = true;
}