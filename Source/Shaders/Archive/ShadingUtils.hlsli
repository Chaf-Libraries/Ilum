#ifndef __SHADINGCOMMON_HLSL__
#define __SHADINGCOMMON_HLSL__

// Bindless Texture Array
Texture2D<float4> Textures[] : register(t0, space0);

/*
0 - PointClamp
1 - PointWarp
2 - BilinearClamp
3 - BilinearWarp
4 - TrilinearClamp
5 - TrilinearWarp
6 - AnisptropicClamp
7 - AnisptropicWarp
*/
SamplerState SamplerStates[] : register(t1, space0);

#endif