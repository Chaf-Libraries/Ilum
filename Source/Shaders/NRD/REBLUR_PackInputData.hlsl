#line 1 "ParseAndMoveShaderParametersToRootConstantBuffer"
cbuffer _RootShaderParameters
{
}

#line 1 "/Plugin/NRD/Private/REBLUR_PackInputData.cs.usf"
#line 1 "/Engine/Public/Platform.ush"
#line 9 "/Engine/Public/Platform.ush"
#line 1 "FP16Math.ush"
#line 10 "/Engine/Public/Platform.ush"
#line 36 "/Engine/Public/Platform.ush"
#line 1 "Platform/D3D/D3DCommon.ush"
#line 10 "/Engine/Public/Platform/D3D/D3DCommon.ush"
precise float MakePrecise(precise float v) { return v; }
precise float2 MakePrecise(precise float2 v) { return v; }
precise float3 MakePrecise(precise float3 v) { return v; }
precise float4 MakePrecise(precise float4 v) { return v; }





SamplerState D3DStaticPointWrappedSampler : register(s0, space1000);
SamplerState D3DStaticPointClampedSampler : register(s1, space1000);
SamplerState D3DStaticBilinearWrappedSampler : register(s2, space1000);
SamplerState D3DStaticBilinearClampedSampler : register(s3, space1000);
SamplerState D3DStaticTrilinearWrappedSampler : register(s4, space1000);
SamplerState D3DStaticTrilinearClampedSampler : register(s5, space1000);
#line 36 "/Engine/Public/Platform/D3D/D3DCommon.ush"
RWStructuredBuffer<uint> UEDiagnosticBuffer : register(u0, space999);

void UEReportAssertWithPayload(uint AssertID, uint4 Payload)
{
	if (WaveIsFirstLane())
	{

		uint OldValue = 0;
		InterlockedAdd(UEDiagnosticBuffer[0], 1, OldValue);
		if (OldValue == 0)
		{
			UEDiagnosticBuffer[1] = AssertID;
			UEDiagnosticBuffer[2] = Payload.x;
			UEDiagnosticBuffer[3] = Payload.y;
			UEDiagnosticBuffer[4] = Payload.z;
			UEDiagnosticBuffer[5] = Payload.w;
		}


		UEDiagnosticBuffer[0xFFFFFFFF] = 0;
	}
}
#line 37 "/Engine/Public/Platform.ush"
#line 42 "/Engine/Public/Platform.ush"
#line 1 "ShaderVersion.ush"
#line 43 "/Engine/Public/Platform.ush"
#line 457 "/Engine/Public/Platform.ush"
void ShaderYield()
{

}
#line 691 "/Engine/Public/Platform.ush"
float min3( float a, float b, float c )
{
	return min( a, min( b, c ) );
}

float max3( float a, float b, float c )
{
	return max( a, max( b, c ) );
}

float2 min3( float2 a, float2 b, float2 c )
{
	return float2(
		min3( a.x, b.x, c.x ),
		min3( a.y, b.y, c.y )
	);
}

float2 max3( float2 a, float2 b, float2 c )
{
	return float2(
		max3( a.x, b.x, c.x ),
		max3( a.y, b.y, c.y )
	);
}

float3 max3( float3 a, float3 b, float3 c )
{
	return float3(
		max3( a.x, b.x, c.x ),
		max3( a.y, b.y, c.y ),
		max3( a.z, b.z, c.z )
	);
}

float3 min3( float3 a, float3 b, float3 c )
{
	return float3(
		min3( a.x, b.x, c.x ),
		min3( a.y, b.y, c.y ),
		min3( a.z, b.z, c.z )
	);
}

float4 min3( float4 a, float4 b, float4 c )
{
	return float4(
		min3( a.x, b.x, c.x ),
		min3( a.y, b.y, c.y ),
		min3( a.z, b.z, c.z ),
		min3( a.w, b.w, c.w )
	);
}

float4 max3( float4 a, float4 b, float4 c )
{
	return float4(
		max3( a.x, b.x, c.x ),
		max3( a.y, b.y, c.y ),
		max3( a.z, b.z, c.z ),
		max3( a.w, b.w, c.w )
	);
}
#line 848 "/Engine/Public/Platform.ush"
float CondMask(bool Cond, float Src0, float Src1) { return Cond ? Src0 : Src1; }
float2 CondMask(bool Cond, float2 Src0, float2 Src1) { return Cond ? Src0 : Src1; }
float3 CondMask(bool Cond, float3 Src0, float3 Src1) { return Cond ? Src0 : Src1; }
float4 CondMask(bool Cond, float4 Src0, float4 Src1) { return Cond ? Src0 : Src1; }

int CondMask(bool Cond, int Src0, int Src1) { return Cond ? Src0 : Src1; }
int2 CondMask(bool Cond, int2 Src0, int2 Src1) { return Cond ? Src0 : Src1; }
int3 CondMask(bool Cond, int3 Src0, int3 Src1) { return Cond ? Src0 : Src1; }
int4 CondMask(bool Cond, int4 Src0, int4 Src1) { return Cond ? Src0 : Src1; }

uint CondMask(bool Cond, uint Src0, uint Src1) { return Cond ? Src0 : Src1; }
uint2 CondMask(bool Cond, uint2 Src0, uint2 Src1) { return Cond ? Src0 : Src1; }
uint3 CondMask(bool Cond, uint3 Src0, uint3 Src1) { return Cond ? Src0 : Src1; }
uint4 CondMask(bool Cond, uint4 Src0, uint4 Src1) { return Cond ? Src0 : Src1; }




float UnpackByte0(uint v) { return float(v & 0xff); }
float UnpackByte1(uint v) { return float((v >> 8) & 0xff); }
float UnpackByte2(uint v) { return float((v >> 16) & 0xff); }
float UnpackByte3(uint v) { return float(v >> 24); }









uint BitFieldInsertU32(uint Mask, uint Preserve, uint Enable)
{
	return (Preserve & Mask) | (Enable & ~Mask);
}

uint BitFieldExtractU32(uint Data, uint Size, uint Offset)
{
	Size &= 31u;
	Offset &= 31u;

	if (Size == 0u)
		return 0u;
	else if (Offset + Size < 32u)
		return (Data << (32u - Size - Offset)) >> (32u - Size);
	else
		return Data >> Offset;
}

int BitFieldExtractI32(int Data, uint Size, uint Offset)
{
	Size &= 31u;
	Offset &= 31u;

	if (Size == 0u)
		return 0;
	else if (Offset + Size < 32u)
		return (Data << (32u - Size - Offset)) >> (32u - Size);
	else
		return Data >> Offset;
}

uint BitFieldMaskU32(uint MaskWidth, uint MaskLocation)
{
	MaskWidth &= 31u;
	MaskLocation &= 31u;

	return ((1u << MaskWidth) - 1u) << MaskLocation;
}




uint BitAlignU32(uint High, uint Low, uint Shift)
{
	Shift &= 31u;

	uint Result = Low >> Shift;
	Result |= Shift > 0u ? (High << (32u - Shift)) : 0u;
	return Result;
}




uint ByteAlignU32(uint High, uint Low, uint Shift)
{
	return BitAlignU32(High, Low, Shift * 8);
}
#line 948 "/Engine/Public/Platform.ush"
uint2  PackUlongType(uint2 Value)
{
	return Value;
}

uint2 UnpackUlongType( uint2  Value)
{
	return Value;
}




uint MaskedBitCount( uint2 Bits, uint Index )
{
	bool bLow = Index < 32;

	uint Mask = 1u << ( Index - ( bLow ? 0 : 32 ) );
	Mask -= 1;

	uint Offset;
	Offset = countbits( Bits.x & ( bLow ? Mask : ~0u ) );
	Offset += countbits( Bits.y & ( bLow ? 0 : Mask ) );
	return Offset;
}










uint MaskedBitCount( uint2 Bits )
{
	return MaskedBitCount( Bits, WaveGetLaneIndex() );
}



uint2 WaveBallot( bool Expr )
{
	return WaveActiveBallot( Expr ).xy;
}



uint WaveGetActiveLaneIndexLast()
{
	uint2 ActiveMask = WaveActiveBallot( true ).xy;
	return firstbithigh( ActiveMask.y ? ActiveMask.y : ActiveMask.x ) + ( ActiveMask.y ? 32 : 0 );
}
#line 2 "/Plugin/NRD/Private/REBLUR_PackInputData.cs.usf"
#line 1 "/Engine/Private/ScreenPass.ush"
#line 26 "/Engine/Private/ScreenPass.ush"
float2 ApplyScreenTransform(float2 PInA,  float4  AToB)
{
	return PInA * AToB.xy + AToB.zw;
}
#line 3 "/Plugin/NRD/Private/REBLUR_PackInputData.cs.usf"
#line 1 "/Engine/Private/SceneTextureParameters.ush"
#line 5 "/Engine/Private/SceneTextureParameters.ush"
#line 1 "SceneTexturesCommon.ush"
#line 9 "/Engine/Private/SceneTexturesCommon.ush"
#line 1 "Common.ush"
#line 31 "/Engine/Private/Common.ush"
struct FloatDeriv
{
	float Value;
	float Ddx;
	float Ddy;
};

struct FloatDeriv2
{
	float2 Value;
	float2 Ddx;
	float2 Ddy;
};

struct FloatDeriv3
{
	float3 Value;
	float3 Ddx;
	float3 Ddy;
};

struct FloatDeriv4
{
	float4 Value;
	float4 Ddx;
	float4 Ddy;
};

FloatDeriv ConstructFloatDeriv(float InValue, float InDdx, float InDdy)
{
	FloatDeriv Ret;
	Ret.Value = InValue;
	Ret.Ddx = InDdx;
	Ret.Ddy = InDdy;
	return Ret;
}

FloatDeriv2 ConstructFloatDeriv2(float2 InValue, float2 InDdx, float2 InDdy)
{
	FloatDeriv2 Ret;
	Ret.Value = InValue;
	Ret.Ddx = InDdx;
	Ret.Ddy = InDdy;
	return Ret;
}

FloatDeriv3 ConstructFloatDeriv3(float3 InValue, float3 InDdx, float3 InDdy)
{
	FloatDeriv3 Ret;
	Ret.Value = InValue;
	Ret.Ddx = InDdx;
	Ret.Ddy = InDdy;
	return Ret;
}

FloatDeriv4 ConstructFloatDeriv4(float4 InValue, float4 InDdx, float4 InDdy)
{
	FloatDeriv4 Ret;
	Ret.Value = InValue;
	Ret.Ddx = InDdx;
	Ret.Ddy = InDdy;
	return Ret;
}
#line 109 "/Engine/Private/Common.ush"
const static  float  PI = 3.1415926535897932f;
const static float MaxHalfFloat = 65504.0f;
const static float Max11BitsFloat = 65024.0f;
const static float Max10BitsFloat = 64512.0f;
const static float3 Max111110BitsFloat3 = float3(Max11BitsFloat, Max11BitsFloat, Max10BitsFloat);
#line 166 "/Engine/Private/Common.ush"
#line 1 "GeneratedUniformBufferTypes.ush"
#line 8 "/Engine/Private/GeneratedUniformBufferTypes.ush"
#line 1 "Nanite/NanitePackedNaniteView.ush"
#line 5 "/Engine/Private/Nanite/NanitePackedNaniteView.ush"
struct FPackedNaniteView
{
	float4x4 SVPositionToTranslatedWorld;
	float4x4 ViewToTranslatedWorld;

	float4x4 TranslatedWorldToView;
	float4x4 TranslatedWorldToClip;
	float4x4 ViewToClip;
	float4x4 ClipToRelativeWorld;

	float4x4 PrevTranslatedWorldToView;
	float4x4 PrevTranslatedWorldToClip;
	float4x4 PrevViewToClip;
	float4x4 PrevClipToRelativeWorld;

	int4 ViewRect;
	float4 ViewSizeAndInvSize;
	float4 ClipSpaceScaleOffset;
	float4 PreViewTranslation;
	float4 PrevPreViewTranslation;
	float4 WorldCameraOrigin;
	float4 ViewForwardAndNearPlane;

	float3 ViewTilePosition;
	float Padding0;

	float3 MatrixTilePosition;
	float Padding1;

	float2 LODScales;
	float MinBoundsRadiusSq;
	uint StreamingPriorityCategory_AndFlags;

	int4 TargetLayerIdX_AndMipLevelY_AndNumMipLevelsZ;

	int4 HZBTestViewRect;
};
#line 9 "/Engine/Private/GeneratedUniformBufferTypes.ush"
#line 11 "/Engine/Private/GeneratedUniformBufferTypes.ush"
#line 1 "HairStrands/HairStrandsVisibilityCommonStruct.ush"
#line 5 "/Engine/Private/HairStrands/HairStrandsVisibilityCommonStruct.ush"
struct FHairMaterialParameterEx
{
	float4 D;
	float4 R;
	float4 TT;
	float4 TRT;
	float4 DualScatter;
	float4 LightingIntensity;
	float4 CustomAO_0;
	float4 CustomAO_1;
};


struct FPPLLNodeData
{
	float Depth;
	uint Specular_LightChannel_Backlit_Coverage16bit;
	uint PrimitiveID_MacroGroupID;
	uint MaterialID;
	uint Tangent_Coverage8bit;
	uint BaseColor_Roughness;
	uint NextNodeIndex;
	uint PackedVelocity;
};

struct FPackedHairSample
{
	float Depth;
	uint PrimitiveID_MacroGroupID;
	uint MaterialID;
	uint Tangent_Coverage8bit;
	uint BaseColor_Roughness;
	uint Specular_LightChannels_Backlit;
};

struct FPackedHairVis
{
	uint Depth_Coverage8bit;
	uint PrimitiveID_MaterialID;
};
#line 12 "/Engine/Private/GeneratedUniformBufferTypes.ush"
#line 1 "HairStrands/HairStrandsVoxelPageCommonStruct.ush"
#line 5 "/Engine/Private/HairStrands/HairStrandsVoxelPageCommonStruct.ush"
struct FPackedVirtualVoxelNodeDesc
{
	float3 TranslatedWorldMinAABB;
	uint PackedPageIndexResolution;
	float3 TranslatedWorldMaxAABB;
	uint PageIndexOffset;
};

struct FVoxelizationViewInfo
{
	float4x4 TranslatedWorldToClip;
	float3 ViewForward;
	float Pad0;
	uint2 RasterResolution;
	float2 Pad1;
};
#line 13 "/Engine/Private/GeneratedUniformBufferTypes.ush"
#line 1 "HairStrands/HairStrandsDeepShadowCommonStruct.ush"
#line 5 "/Engine/Private/HairStrands/HairStrandsDeepShadowCommonStruct.ush"
struct FDeepShadowViewInfo
{
	float4x4 TranslatedWorldToClip;
	float3 ViewForward;
	float MinRadiusAtDepth1;
};

struct FDeepShadowTransform
{
	float4x4 TranslatedWorldToClip;
};
#line 14 "/Engine/Private/GeneratedUniformBufferTypes.ush"
#line 167 "/Engine/Private/Common.ush"
#line 169 "/Engine/Private/Common.ush"
#line 1 "/Engine/Generated/GeneratedUniformBuffers.ush"
#line 1 "/Engine/Generated/UniformBuffers/MobileSceneTextures.ush"


cbuffer MobileSceneTextures
{
}
Texture2D MobileSceneTextures_SceneColorTexture;
SamplerState MobileSceneTextures_SceneColorTextureSampler;
Texture2D MobileSceneTextures_SceneDepthTexture;
SamplerState MobileSceneTextures_SceneDepthTextureSampler;
Texture2D MobileSceneTextures_CustomDepthTexture;
SamplerState MobileSceneTextures_CustomDepthTextureSampler;
Texture2D MobileSceneTextures_MobileCustomStencilTexture;
SamplerState MobileSceneTextures_MobileCustomStencilTextureSampler;
Texture2D MobileSceneTextures_SceneVelocityTexture;
SamplerState MobileSceneTextures_SceneVelocityTextureSampler;
Texture2D MobileSceneTextures_GBufferATexture;
Texture2D MobileSceneTextures_GBufferBTexture;
Texture2D MobileSceneTextures_GBufferCTexture;
Texture2D MobileSceneTextures_GBufferDTexture;
Texture2D MobileSceneTextures_SceneDepthAuxTexture;
SamplerState MobileSceneTextures_GBufferATextureSampler;
SamplerState MobileSceneTextures_GBufferBTextureSampler;
SamplerState MobileSceneTextures_GBufferCTextureSampler;
SamplerState MobileSceneTextures_GBufferDTextureSampler;
SamplerState MobileSceneTextures_SceneDepthAuxTextureSampler;
/*atic const struct
{
	Texture2D SceneColorTexture;
	SamplerState SceneColorTextureSampler;
	Texture2D SceneDepthTexture;
	SamplerState SceneDepthTextureSampler;
	Texture2D CustomDepthTexture;
	SamplerState CustomDepthTextureSampler;
	Texture2D MobileCustomStencilTexture;
	SamplerState MobileCustomStencilTextureSampler;
	Texture2D SceneVelocityTexture;
	SamplerState SceneVelocityTextureSampler;
	Texture2D GBufferATexture;
	Texture2D GBufferBTexture;
	Texture2D GBufferCTexture;
	Texture2D GBufferDTexture;
	Texture2D SceneDepthAuxTexture;
	SamplerState GBufferATextureSampler;
	SamplerState GBufferBTextureSampler;
	SamplerState GBufferCTextureSampler;
	SamplerState GBufferDTextureSampler;
	SamplerState SceneDepthAuxTextureSampler;
} MobileSceneTextures = {MobileSceneTextures_SceneColorTexture,MobileSceneTextures_SceneColorTextureSampler,MobileSceneTextures_SceneDepthTexture,MobileSceneTextures_SceneDepthTextureSampler,MobileSceneTextures_CustomDepthTexture,MobileSceneTextures_CustomDepthTextureSampler,MobileSceneTextures_MobileCustomStencilTexture,MobileSceneTextures_MobileCustomStencilTextureSampler,MobileSceneTextures_SceneVelocityTexture,MobileSceneTextures_SceneVelocityTextureSampler,MobileSceneTextures_GBufferATexture,MobileSceneTextures_GBufferBTexture,MobileSceneTextures_GBufferCTexture,MobileSceneTextures_GBufferDTexture,MobileSceneTextures_SceneDepthAuxTexture,MobileSceneTextures_GBufferATextureSampler,MobileSceneTextures_GBufferBTextureSampler,MobileSceneTextures_GBufferCTextureSampler,MobileSceneTextures_GBufferDTextureSampler,MobileSceneTextures_SceneDepthAuxTextureSampler,*/
#line 2 "/Engine/Generated/GeneratedUniformBuffers.ush"
#line 1 "/Engine/Generated/UniformBuffers/SceneTexturesStruct.ush"


cbuffer SceneTexturesStruct
{
}
Texture2D SceneTexturesStruct_SceneColorTexture;
Texture2D SceneTexturesStruct_SceneDepthTexture;
Texture2D SceneTexturesStruct_GBufferATexture;
Texture2D SceneTexturesStruct_GBufferBTexture;
Texture2D SceneTexturesStruct_GBufferCTexture;
Texture2D SceneTexturesStruct_GBufferDTexture;
Texture2D SceneTexturesStruct_GBufferETexture;
Texture2D SceneTexturesStruct_GBufferFTexture;
Texture2D SceneTexturesStruct_GBufferVelocityTexture;
Texture2D SceneTexturesStruct_ScreenSpaceAOTexture;
Texture2D SceneTexturesStruct_CustomDepthTexture;
Texture2D<uint2> SceneTexturesStruct_CustomStencilTexture;
SamplerState SceneTexturesStruct_PointClampSampler;
/*atic const struct
{
	Texture2D SceneColorTexture;
	Texture2D SceneDepthTexture;
	Texture2D GBufferATexture;
	Texture2D GBufferBTexture;
	Texture2D GBufferCTexture;
	Texture2D GBufferDTexture;
	Texture2D GBufferETexture;
	Texture2D GBufferFTexture;
	Texture2D GBufferVelocityTexture;
	Texture2D ScreenSpaceAOTexture;
	Texture2D CustomDepthTexture;
	Texture2D<uint2> CustomStencilTexture;
	SamplerState PointClampSampler;
} SceneTexturesStruct = {SceneTexturesStruct_SceneColorTexture,SceneTexturesStruct_SceneDepthTexture,SceneTexturesStruct_GBufferATexture,SceneTexturesStruct_GBufferBTexture,SceneTexturesStruct_GBufferCTexture,SceneTexturesStruct_GBufferDTexture,SceneTexturesStruct_GBufferETexture,SceneTexturesStruct_GBufferFTexture,SceneTexturesStruct_GBufferVelocityTexture,SceneTexturesStruct_ScreenSpaceAOTexture,SceneTexturesStruct_CustomDepthTexture,SceneTexturesStruct_CustomStencilTexture,SceneTexturesStruct_PointClampSampler,*/
#line 3 "/Engine/Generated/GeneratedUniformBuffers.ush"
#line 1 "/Engine/Generated/UniformBuffers/View.ush"


cbuffer View
{
	float4x4 View_TranslatedWorldToClip;
	float4x4 View_RelativeWorldToClip;
	float4x4 View_ClipToRelativeWorld;
	float4x4 View_TranslatedWorldToView;
	float4x4 View_ViewToTranslatedWorld;
	float4x4 View_TranslatedWorldToCameraView;
	float4x4 View_CameraViewToTranslatedWorld;
	float4x4 View_ViewToClip;
	float4x4 View_ViewToClipNoAA;
	float4x4 View_ClipToView;
	float4x4 View_ClipToTranslatedWorld;
	float4x4 View_SVPositionToTranslatedWorld;
	float4x4 View_ScreenToRelativeWorld;
	float4x4 View_ScreenToTranslatedWorld;
	float4x4 View_MobileMultiviewShadowTransform;
	float3 View_ViewTilePosition;
	float PrePadding_View_972;
	float3 View_MatrixTilePosition;
	float PrePadding_View_988;
	float3 View_ViewForward;
	float PrePadding_View_1004;
	float3 View_ViewUp;
	float PrePadding_View_1020;
	float3 View_ViewRight;
	float PrePadding_View_1036;
	float3 View_HMDViewNoRollUp;
	float PrePadding_View_1052;
	float3 View_HMDViewNoRollRight;
	float PrePadding_View_1068;
	float4 View_InvDeviceZToWorldZTransform;
	float4 View_ScreenPositionScaleBias;
	float3 View_RelativeWorldCameraOrigin;
	float PrePadding_View_1116;
	float3 View_TranslatedWorldCameraOrigin;
	float PrePadding_View_1132;
	float3 View_RelativeWorldViewOrigin;
	float PrePadding_View_1148;
	float3 View_RelativePreViewTranslation;
	float PrePadding_View_1164;
	float4x4 View_PrevViewToClip;
	float4x4 View_PrevClipToView;
	float4x4 View_PrevTranslatedWorldToClip;
	float4x4 View_PrevTranslatedWorldToView;
	float4x4 View_PrevViewToTranslatedWorld;
	float4x4 View_PrevTranslatedWorldToCameraView;
	float4x4 View_PrevCameraViewToTranslatedWorld;
	float3 View_PrevTranslatedWorldCameraOrigin;
	float PrePadding_View_1628;
	float3 View_PrevRelativeWorldCameraOrigin;
	float PrePadding_View_1644;
	float3 View_PrevRelativeWorldViewOrigin;
	float PrePadding_View_1660;
	float3 View_RelativePrevPreViewTranslation;
	float PrePadding_View_1676;
	float4x4 View_PrevClipToRelativeWorld;
	float4x4 View_PrevScreenToTranslatedWorld;
	float4x4 View_ClipToPrevClip;
	float4x4 View_ClipToPrevClipWithAA;
	float4 View_TemporalAAJitter;
	float4 View_GlobalClippingPlane;
	float2 View_FieldOfViewWideAngles;
	float2 View_PrevFieldOfViewWideAngles;
	float4 View_ViewRectMin;
	float4 View_ViewSizeAndInvSize;
	float4 View_LightProbeSizeRatioAndInvSizeRatio;
	float4 View_BufferSizeAndInvSize;
	float4 View_BufferBilinearUVMinMax;
	float4 View_ScreenToViewSpace;
	int View_NumSceneColorMSAASamples;
	float View_PreExposure;
	float View_OneOverPreExposure;
	float PrePadding_View_2092;
	float4 View_DiffuseOverrideParameter;
	float4 View_SpecularOverrideParameter;
	float4 View_NormalOverrideParameter;
	float2 View_RoughnessOverrideParameter;
	float View_PrevFrameGameTime;
	float View_PrevFrameRealTime;
	float View_OutOfBoundsMask;
	float PrePadding_View_2164;
	float PrePadding_View_2168;
	float PrePadding_View_2172;
	float3 View_WorldCameraMovementSinceLastFrame;
	float View_CullingSign;
	float View_NearPlane;
	float View_GameTime;
	float View_RealTime;
	float View_DeltaTime;
	float View_MaterialTextureMipBias;
	float View_MaterialTextureDerivativeMultiply;
	uint View_Random;
	uint View_FrameNumber;
	uint View_StateFrameIndexMod8;
	uint View_StateFrameIndex;
	uint View_StateRawFrameIndex;
	uint PrePadding_View_2236;
	float4 View_AntiAliasingSampleParams;
	uint View_DebugViewModeMask;
	uint View_DebugInput0;
	float View_CameraCut;
	float View_UnlitViewmodeMask;
	float4 View_DirectionalLightColor;
	float3 View_DirectionalLightDirection;
	float PrePadding_View_2300;
	float4 View_TranslucencyLightingVolumeMin[2];
	float4 View_TranslucencyLightingVolumeInvSize[2];
	float4 View_TemporalAAParams;
	float4 View_CircleDOFParams;
	uint View_ForceDrawAllVelocities;
	float View_DepthOfFieldSensorWidth;
	float View_DepthOfFieldFocalDistance;
	float View_DepthOfFieldScale;
	float View_DepthOfFieldFocalLength;
	float View_DepthOfFieldFocalRegion;
	float View_DepthOfFieldNearTransitionRegion;
	float View_DepthOfFieldFarTransitionRegion;
	float View_MotionBlurNormalizedToPixel;
	float View_GeneralPurposeTweak;
	float View_GeneralPurposeTweak2;
	float View_DemosaicVposOffset;
	float View_DecalDepthBias;
	float PrePadding_View_2452;
	float PrePadding_View_2456;
	float PrePadding_View_2460;
	float3 View_IndirectLightingColorScale;
	float PrePadding_View_2476;
	float3 View_PrecomputedIndirectLightingColorScale;
	float PrePadding_View_2492;
	float3 View_PrecomputedIndirectSpecularColorScale;
	float PrePadding_View_2508;
	float4 View_AtmosphereLightDirection[2];
	float4 View_AtmosphereLightIlluminanceOnGroundPostTransmittance[2];
	float4 View_AtmosphereLightIlluminanceOuterSpace[2];
	float4 View_AtmosphereLightDiscLuminance[2];
	float4 View_AtmosphereLightDiscCosHalfApexAngle[2];
	float4 View_SkyViewLutSizeAndInvSize;
	float3 View_SkyCameraTranslatedWorldOrigin;
	float PrePadding_View_2700;
	float4 View_SkyPlanetTranslatedWorldCenterAndViewHeight;
	float4x4 View_SkyViewLutReferential;
	float4 View_SkyAtmosphereSkyLuminanceFactor;
	float View_SkyAtmospherePresentInScene;
	float View_SkyAtmosphereHeightFogContribution;
	float View_SkyAtmosphereBottomRadiusKm;
	float View_SkyAtmosphereTopRadiusKm;
	float4 View_SkyAtmosphereCameraAerialPerspectiveVolumeSizeAndInvSize;
	float View_SkyAtmosphereAerialPerspectiveStartDepthKm;
	float View_SkyAtmosphereCameraAerialPerspectiveVolumeDepthResolution;
	float View_SkyAtmosphereCameraAerialPerspectiveVolumeDepthResolutionInv;
	float View_SkyAtmosphereCameraAerialPerspectiveVolumeDepthSliceLengthKm;
	float View_SkyAtmosphereCameraAerialPerspectiveVolumeDepthSliceLengthKmInv;
	float View_SkyAtmosphereApplyCameraAerialPerspectiveVolume;
	float PrePadding_View_2856;
	float PrePadding_View_2860;
	float3 View_NormalCurvatureToRoughnessScaleBias;
	float View_RenderingReflectionCaptureMask;
	float View_RealTimeReflectionCapture;
	float View_RealTimeReflectionCapturePreExposure;
	float PrePadding_View_2888;
	float PrePadding_View_2892;
	float4 View_AmbientCubemapTint;
	float View_AmbientCubemapIntensity;
	float View_SkyLightApplyPrecomputedBentNormalShadowingFlag;
	float View_SkyLightAffectReflectionFlag;
	float View_SkyLightAffectGlobalIlluminationFlag;
	float4 View_SkyLightColor;
	float4 View_MobileSkyIrradianceEnvironmentMap[7];
	float View_MobilePreviewMode;
	float View_HMDEyePaddingOffset;
	float View_ReflectionCubemapMaxMip;
	float View_ShowDecalsMask;
	uint View_DistanceFieldAOSpecularOcclusionMode;
	float View_IndirectCapsuleSelfShadowingIntensity;
	float PrePadding_View_3080;
	float PrePadding_View_3084;
	float3 View_ReflectionEnvironmentRoughnessMixingScaleBiasAndLargestWeight;
	int View_StereoPassIndex;
	float4 View_GlobalVolumeCenterAndExtent[6];
	float4 View_GlobalVolumeWorldToUVAddAndMul[6];
	float4 View_GlobalDistanceFieldMipWorldToUVScale[6];
	float4 View_GlobalDistanceFieldMipWorldToUVBias[6];
	float View_GlobalDistanceFieldMipFactor;
	float View_GlobalDistanceFieldMipTransition;
	int View_GlobalDistanceFieldClipmapSizeInPages;
	int PrePadding_View_3500;
	float3 View_GlobalDistanceFieldInvPageAtlasSize;
	float PrePadding_View_3516;
	float3 View_GlobalDistanceFieldInvCoverageAtlasSize;
	float View_GlobalVolumeDimension;
	float View_GlobalVolumeTexelSize;
	float View_MaxGlobalDFAOConeDistance;
	uint View_NumGlobalSDFClipmaps;
	float View_FullyCoveredExpandSurfaceScale;
	float View_UncoveredExpandSurfaceScale;
	float View_UncoveredMinStepScale;
	int2 View_CursorPosition;
	float View_bCheckerboardSubsurfaceProfileRendering;
	float PrePadding_View_3572;
	float PrePadding_View_3576;
	float PrePadding_View_3580;
	float3 View_VolumetricFogInvGridSize;
	float PrePadding_View_3596;
	float3 View_VolumetricFogGridZParams;
	float PrePadding_View_3612;
	float2 View_VolumetricFogSVPosToVolumeUV;
	float View_VolumetricFogMaxDistance;
	float PrePadding_View_3628;
	float3 View_VolumetricLightmapWorldToUVScale;
	float PrePadding_View_3644;
	float3 View_VolumetricLightmapWorldToUVAdd;
	float PrePadding_View_3660;
	float3 View_VolumetricLightmapIndirectionTextureSize;
	float View_VolumetricLightmapBrickSize;
	float3 View_VolumetricLightmapBrickTexelSize;
	float View_StereoIPD;
	float View_IndirectLightingCacheShowFlag;
	float View_EyeToPixelSpreadAngle;
	float PrePadding_View_3704;
	float PrePadding_View_3708;
	float4 View_XRPassthroughCameraUVs[2];
	float View_GlobalVirtualTextureMipBias;
	uint View_VirtualTextureFeedbackShift;
	uint View_VirtualTextureFeedbackMask;
	uint View_VirtualTextureFeedbackStride;
	uint View_VirtualTextureFeedbackJitterOffset;
	uint View_VirtualTextureFeedbackSampleOffset;
	uint PrePadding_View_3768;
	uint PrePadding_View_3772;
	float4 View_RuntimeVirtualTextureMipLevel;
	float2 View_RuntimeVirtualTexturePackHeight;
	float PrePadding_View_3800;
	float PrePadding_View_3804;
	float4 View_RuntimeVirtualTextureDebugParams;
	float View_OverrideLandscapeLOD;
	int View_FarShadowStaticMeshLODBias;
	float View_MinRoughness;
	float PrePadding_View_3836;
	float4 View_HairRenderInfo;
	uint View_EnableSkyLight;
	uint View_HairRenderInfoBits;
	uint View_HairComponents;
	float View_bSubsurfacePostprocessEnabled;
	float4 View_SSProfilesTextureSizeAndInvSize;
	float4 View_SSProfilesPreIntegratedTextureSizeAndInvSize;
	float3 View_PhysicsFieldClipmapCenter;
	float View_PhysicsFieldClipmapDistance;
	int View_PhysicsFieldClipmapResolution;
	int View_PhysicsFieldClipmapExponent;
	int View_PhysicsFieldClipmapCount;
	int View_PhysicsFieldTargetCount;
	int4 View_PhysicsFieldTargets[32];
	uint View_InstanceSceneDataSOAStride;
	uint View_GPUSceneViewId;
	uint PrePadding_View_4456;
	uint PrePadding_View_4460;
	uint PrePadding_View_4464;
	uint PrePadding_View_4468;
	uint PrePadding_View_4472;
	uint PrePadding_View_4476;
	uint PrePadding_View_4480;
	uint PrePadding_View_4484;
	uint PrePadding_View_4488;
	uint PrePadding_View_4492;
	uint PrePadding_View_4496;
	uint PrePadding_View_4500;
	uint PrePadding_View_4504;
	uint PrePadding_View_4508;
	uint PrePadding_View_4512;
	uint PrePadding_View_4516;
	uint PrePadding_View_4520;
	uint PrePadding_View_4524;
	uint PrePadding_View_4528;
	uint PrePadding_View_4532;
	uint PrePadding_View_4536;
	uint PrePadding_View_4540;
	uint PrePadding_View_4544;
	uint PrePadding_View_4548;
	uint PrePadding_View_4552;
	uint PrePadding_View_4556;
	uint PrePadding_View_4560;
	uint PrePadding_View_4564;
	uint PrePadding_View_4568;
	uint PrePadding_View_4572;
	uint PrePadding_View_4576;
	uint PrePadding_View_4580;
	uint PrePadding_View_4584;
	uint PrePadding_View_4588;
	uint PrePadding_View_4592;
	uint PrePadding_View_4596;
	uint PrePadding_View_4600;
	uint PrePadding_View_4604;
	uint PrePadding_View_4608;
	uint PrePadding_View_4612;
	uint PrePadding_View_4616;
	uint PrePadding_View_4620;
	uint PrePadding_View_4624;
	uint PrePadding_View_4628;
	uint PrePadding_View_4632;
	uint PrePadding_View_4636;
	uint PrePadding_View_4640;
	uint PrePadding_View_4644;
	uint PrePadding_View_4648;
	uint PrePadding_View_4652;
	uint PrePadding_View_4656;
	uint PrePadding_View_4660;
	uint PrePadding_View_4664;
	uint PrePadding_View_4668;
	uint PrePadding_View_4672;
	uint PrePadding_View_4676;
	uint PrePadding_View_4680;
	uint PrePadding_View_4684;
	uint PrePadding_View_4688;
	uint PrePadding_View_4692;
	uint PrePadding_View_4696;
	uint PrePadding_View_4700;
	uint PrePadding_View_4704;
	uint PrePadding_View_4708;
	uint PrePadding_View_4712;
	uint PrePadding_View_4716;
	uint PrePadding_View_4720;
	uint PrePadding_View_4724;
	uint PrePadding_View_4728;
	uint PrePadding_View_4732;
	uint PrePadding_View_4736;
	uint PrePadding_View_4740;
	uint PrePadding_View_4744;
	uint PrePadding_View_4748;
	uint PrePadding_View_4752;
	uint PrePadding_View_4756;
	uint PrePadding_View_4760;
	uint PrePadding_View_4764;
	uint PrePadding_View_4768;
	uint PrePadding_View_4772;
	uint PrePadding_View_4776;
	uint PrePadding_View_4780;
	uint PrePadding_View_4784;
	uint PrePadding_View_4788;
	uint PrePadding_View_4792;
	uint PrePadding_View_4796;
	uint PrePadding_View_4800;
	uint PrePadding_View_4804;
	uint PrePadding_View_4808;
	uint PrePadding_View_4812;
	uint PrePadding_View_4816;
	uint PrePadding_View_4820;
	uint PrePadding_View_4824;
	uint PrePadding_View_4828;
	uint PrePadding_View_4832;
	uint PrePadding_View_4836;
	uint PrePadding_View_4840;
	uint PrePadding_View_4844;
	uint PrePadding_View_4848;
	uint PrePadding_View_4852;
	uint PrePadding_View_4856;
	uint PrePadding_View_4860;
	uint PrePadding_View_4864;
	uint PrePadding_View_4868;
	uint PrePadding_View_4872;
	uint PrePadding_View_4876;
	uint PrePadding_View_4880;
	uint PrePadding_View_4884;
	uint PrePadding_View_4888;
	uint PrePadding_View_4892;
	uint PrePadding_View_4896;
	uint PrePadding_View_4900;
	uint PrePadding_View_4904;
	uint PrePadding_View_4908;
	uint PrePadding_View_4912;
	uint PrePadding_View_4916;
	uint PrePadding_View_4920;
	uint PrePadding_View_4924;
	uint PrePadding_View_4928;
	uint PrePadding_View_4932;
	uint PrePadding_View_4936;
	uint PrePadding_View_4940;
	uint PrePadding_View_4944;
	uint PrePadding_View_4948;
	uint PrePadding_View_4952;
	uint PrePadding_View_4956;
	uint PrePadding_View_4960;
	uint PrePadding_View_4964;
	uint PrePadding_View_4968;
	uint PrePadding_View_4972;
	uint PrePadding_View_4976;
	uint PrePadding_View_4980;
	uint View_bShadingEnergyConservation;
	uint View_bShadingEnergyPreservation;
}
SamplerState View_MaterialTextureBilinearWrapedSampler;
SamplerState View_MaterialTextureBilinearClampedSampler;
Texture3D<uint4> View_VolumetricLightmapIndirectionTexture;
Texture3D View_VolumetricLightmapBrickAmbientVector;
Texture3D View_VolumetricLightmapBrickSHCoefficients0;
Texture3D View_VolumetricLightmapBrickSHCoefficients1;
Texture3D View_VolumetricLightmapBrickSHCoefficients2;
Texture3D View_VolumetricLightmapBrickSHCoefficients3;
Texture3D View_VolumetricLightmapBrickSHCoefficients4;
Texture3D View_VolumetricLightmapBrickSHCoefficients5;
Texture3D View_SkyBentNormalBrickTexture;
Texture3D View_DirectionalLightShadowingBrickTexture;
SamplerState View_VolumetricLightmapBrickAmbientVectorSampler;
SamplerState View_VolumetricLightmapTextureSampler0;
SamplerState View_VolumetricLightmapTextureSampler1;
SamplerState View_VolumetricLightmapTextureSampler2;
SamplerState View_VolumetricLightmapTextureSampler3;
SamplerState View_VolumetricLightmapTextureSampler4;
SamplerState View_VolumetricLightmapTextureSampler5;
SamplerState View_SkyBentNormalTextureSampler;
SamplerState View_DirectionalLightShadowingTextureSampler;
Texture3D View_GlobalDistanceFieldPageAtlasTexture;
Texture3D View_GlobalDistanceFieldCoverageAtlasTexture;
Texture3D<uint> View_GlobalDistanceFieldPageTableTexture;
Texture3D View_GlobalDistanceFieldMipTexture;
Texture2D View_AtmosphereTransmittanceTexture;
SamplerState View_AtmosphereTransmittanceTextureSampler;
Texture2D View_AtmosphereIrradianceTexture;
SamplerState View_AtmosphereIrradianceTextureSampler;
Texture3D View_AtmosphereInscatterTexture;
SamplerState View_AtmosphereInscatterTextureSampler;
Texture2D View_PerlinNoiseGradientTexture;
SamplerState View_PerlinNoiseGradientTextureSampler;
Texture3D View_PerlinNoise3DTexture;
SamplerState View_PerlinNoise3DTextureSampler;
Texture2D<uint> View_SobolSamplingTexture;
Texture2D View_BNDSequence_OwenScrambledSequence;
Texture2D View_BNDSequence_RankingScramblingTile;
SamplerState View_SharedPointWrappedSampler;
SamplerState View_SharedPointClampedSampler;
SamplerState View_SharedBilinearWrappedSampler;
SamplerState View_SharedBilinearClampedSampler;
SamplerState View_SharedBilinearAnisoClampedSampler;
SamplerState View_SharedTrilinearWrappedSampler;
SamplerState View_SharedTrilinearClampedSampler;
Texture2D View_PreIntegratedBRDF;
SamplerState View_PreIntegratedBRDFSampler;
StructuredBuffer<float4> View_PrimitiveSceneData;
StructuredBuffer<float4> View_InstanceSceneData;
StructuredBuffer<float4> View_InstancePayloadData;
StructuredBuffer<float4> View_LightmapSceneData;
StructuredBuffer<float4> View_SkyIrradianceEnvironmentMap;
Texture2D View_TransmittanceLutTexture;
SamplerState View_TransmittanceLutTextureSampler;
Texture2D View_SkyViewLutTexture;
SamplerState View_SkyViewLutTextureSampler;
Texture2D View_DistantSkyLightLutTexture;
SamplerState View_DistantSkyLightLutTextureSampler;
Texture3D View_CameraAerialPerspectiveVolume;
SamplerState View_CameraAerialPerspectiveVolumeSampler;
Texture3D View_HairScatteringLUTTexture;
SamplerState View_HairScatteringLUTSampler;
Texture2D View_LTCMatTexture;
SamplerState View_LTCMatSampler;
Texture2D View_LTCAmpTexture;
SamplerState View_LTCAmpSampler;
Texture2D<float2> View_ShadingEnergyGGXSpecTexture;
Texture3D<float2> View_ShadingEnergyGGXGlassTexture;
Texture2D<float2> View_ShadingEnergyClothSpecTexture;
Texture2D<float> View_ShadingEnergyDiffuseTexture;
SamplerState View_ShadingEnergySampler;
Texture2D View_SSProfilesTexture;
SamplerState View_SSProfilesSampler;
SamplerState View_SSProfilesTransmissionSampler;
Texture2DArray View_SSProfilesPreIntegratedTexture;
SamplerState View_SSProfilesPreIntegratedSampler;
Buffer<float4> View_WaterIndirection;
Buffer<float4> View_WaterData;
Buffer<uint> View_LandscapeIndirection;
Buffer<float> View_LandscapePerComponentData;
RWBuffer<uint> View_VTFeedbackBuffer;
Buffer<uint> View_EditorVisualizeLevelInstanceIds;
Buffer<uint> View_EditorSelectedHitProxyIds;
Buffer<float> View_PhysicsFieldClipmapBuffer;
/*atic const struct
{
	float4x4 TranslatedWorldToClip;
	float4x4 RelativeWorldToClip;
	float4x4 ClipToRelativeWorld;
	float4x4 TranslatedWorldToView;
	float4x4 ViewToTranslatedWorld;
	float4x4 TranslatedWorldToCameraView;
	float4x4 CameraViewToTranslatedWorld;
	float4x4 ViewToClip;
	float4x4 ViewToClipNoAA;
	float4x4 ClipToView;
	float4x4 ClipToTranslatedWorld;
	float4x4 SVPositionToTranslatedWorld;
	float4x4 ScreenToRelativeWorld;
	float4x4 ScreenToTranslatedWorld;
	float4x4 MobileMultiviewShadowTransform;
	float3 ViewTilePosition;
	float3 MatrixTilePosition;
	float3 ViewForward;
	float3 ViewUp;
	float3 ViewRight;
	float3 HMDViewNoRollUp;
	float3 HMDViewNoRollRight;
	float4 InvDeviceZToWorldZTransform;
	float4 ScreenPositionScaleBias;
	float3 RelativeWorldCameraOrigin;
	float3 TranslatedWorldCameraOrigin;
	float3 RelativeWorldViewOrigin;
	float3 RelativePreViewTranslation;
	float4x4 PrevViewToClip;
	float4x4 PrevClipToView;
	float4x4 PrevTranslatedWorldToClip;
	float4x4 PrevTranslatedWorldToView;
	float4x4 PrevViewToTranslatedWorld;
	float4x4 PrevTranslatedWorldToCameraView;
	float4x4 PrevCameraViewToTranslatedWorld;
	float3 PrevTranslatedWorldCameraOrigin;
	float3 PrevRelativeWorldCameraOrigin;
	float3 PrevRelativeWorldViewOrigin;
	float3 RelativePrevPreViewTranslation;
	float4x4 PrevClipToRelativeWorld;
	float4x4 PrevScreenToTranslatedWorld;
	float4x4 ClipToPrevClip;
	float4x4 ClipToPrevClipWithAA;
	float4 TemporalAAJitter;
	float4 GlobalClippingPlane;
	float2 FieldOfViewWideAngles;
	float2 PrevFieldOfViewWideAngles;
	float4 ViewRectMin;
	float4 ViewSizeAndInvSize;
	float4 LightProbeSizeRatioAndInvSizeRatio;
	float4 BufferSizeAndInvSize;
	float4 BufferBilinearUVMinMax;
	float4 ScreenToViewSpace;
	int NumSceneColorMSAASamples;
	float PreExposure;
	float OneOverPreExposure;
	float4 DiffuseOverrideParameter;
	float4 SpecularOverrideParameter;
	float4 NormalOverrideParameter;
	float2 RoughnessOverrideParameter;
	float PrevFrameGameTime;
	float PrevFrameRealTime;
	float OutOfBoundsMask;
	float3 WorldCameraMovementSinceLastFrame;
	float CullingSign;
	float NearPlane;
	float GameTime;
	float RealTime;
	float DeltaTime;
	float MaterialTextureMipBias;
	float MaterialTextureDerivativeMultiply;
	uint Random;
	uint FrameNumber;
	uint StateFrameIndexMod8;
	uint StateFrameIndex;
	uint StateRawFrameIndex;
	float4 AntiAliasingSampleParams;
	uint DebugViewModeMask;
	uint DebugInput0;
	float CameraCut;
	float UnlitViewmodeMask;
	float4 DirectionalLightColor;
	float3 DirectionalLightDirection;
	float4 TranslucencyLightingVolumeMin[2];
	float4 TranslucencyLightingVolumeInvSize[2];
	float4 TemporalAAParams;
	float4 CircleDOFParams;
	uint ForceDrawAllVelocities;
	float DepthOfFieldSensorWidth;
	float DepthOfFieldFocalDistance;
	float DepthOfFieldScale;
	float DepthOfFieldFocalLength;
	float DepthOfFieldFocalRegion;
	float DepthOfFieldNearTransitionRegion;
	float DepthOfFieldFarTransitionRegion;
	float MotionBlurNormalizedToPixel;
	float GeneralPurposeTweak;
	float GeneralPurposeTweak2;
	float DemosaicVposOffset;
	float DecalDepthBias;
	float3 IndirectLightingColorScale;
	float3 PrecomputedIndirectLightingColorScale;
	float3 PrecomputedIndirectSpecularColorScale;
	float4 AtmosphereLightDirection[2];
	float4 AtmosphereLightIlluminanceOnGroundPostTransmittance[2];
	float4 AtmosphereLightIlluminanceOuterSpace[2];
	float4 AtmosphereLightDiscLuminance[2];
	float4 AtmosphereLightDiscCosHalfApexAngle[2];
	float4 SkyViewLutSizeAndInvSize;
	float3 SkyCameraTranslatedWorldOrigin;
	float4 SkyPlanetTranslatedWorldCenterAndViewHeight;
	float4x4 SkyViewLutReferential;
	float4 SkyAtmosphereSkyLuminanceFactor;
	float SkyAtmospherePresentInScene;
	float SkyAtmosphereHeightFogContribution;
	float SkyAtmosphereBottomRadiusKm;
	float SkyAtmosphereTopRadiusKm;
	float4 SkyAtmosphereCameraAerialPerspectiveVolumeSizeAndInvSize;
	float SkyAtmosphereAerialPerspectiveStartDepthKm;
	float SkyAtmosphereCameraAerialPerspectiveVolumeDepthResolution;
	float SkyAtmosphereCameraAerialPerspectiveVolumeDepthResolutionInv;
	float SkyAtmosphereCameraAerialPerspectiveVolumeDepthSliceLengthKm;
	float SkyAtmosphereCameraAerialPerspectiveVolumeDepthSliceLengthKmInv;
	float SkyAtmosphereApplyCameraAerialPerspectiveVolume;
	float3 NormalCurvatureToRoughnessScaleBias;
	float RenderingReflectionCaptureMask;
	float RealTimeReflectionCapture;
	float RealTimeReflectionCapturePreExposure;
	float4 AmbientCubemapTint;
	float AmbientCubemapIntensity;
	float SkyLightApplyPrecomputedBentNormalShadowingFlag;
	float SkyLightAffectReflectionFlag;
	float SkyLightAffectGlobalIlluminationFlag;
	float4 SkyLightColor;
	float4 MobileSkyIrradianceEnvironmentMap[7];
	float MobilePreviewMode;
	float HMDEyePaddingOffset;
	float ReflectionCubemapMaxMip;
	float ShowDecalsMask;
	uint DistanceFieldAOSpecularOcclusionMode;
	float IndirectCapsuleSelfShadowingIntensity;
	float3 ReflectionEnvironmentRoughnessMixingScaleBiasAndLargestWeight;
	int StereoPassIndex;
	float4 GlobalVolumeCenterAndExtent[6];
	float4 GlobalVolumeWorldToUVAddAndMul[6];
	float4 GlobalDistanceFieldMipWorldToUVScale[6];
	float4 GlobalDistanceFieldMipWorldToUVBias[6];
	float GlobalDistanceFieldMipFactor;
	float GlobalDistanceFieldMipTransition;
	int GlobalDistanceFieldClipmapSizeInPages;
	float3 GlobalDistanceFieldInvPageAtlasSize;
	float3 GlobalDistanceFieldInvCoverageAtlasSize;
	float GlobalVolumeDimension;
	float GlobalVolumeTexelSize;
	float MaxGlobalDFAOConeDistance;
	uint NumGlobalSDFClipmaps;
	float FullyCoveredExpandSurfaceScale;
	float UncoveredExpandSurfaceScale;
	float UncoveredMinStepScale;
	int2 CursorPosition;
	float bCheckerboardSubsurfaceProfileRendering;
	float3 VolumetricFogInvGridSize;
	float3 VolumetricFogGridZParams;
	float2 VolumetricFogSVPosToVolumeUV;
	float VolumetricFogMaxDistance;
	float3 VolumetricLightmapWorldToUVScale;
	float3 VolumetricLightmapWorldToUVAdd;
	float3 VolumetricLightmapIndirectionTextureSize;
	float VolumetricLightmapBrickSize;
	float3 VolumetricLightmapBrickTexelSize;
	float StereoIPD;
	float IndirectLightingCacheShowFlag;
	float EyeToPixelSpreadAngle;
	float4 XRPassthroughCameraUVs[2];
	float GlobalVirtualTextureMipBias;
	uint VirtualTextureFeedbackShift;
	uint VirtualTextureFeedbackMask;
	uint VirtualTextureFeedbackStride;
	uint VirtualTextureFeedbackJitterOffset;
	uint VirtualTextureFeedbackSampleOffset;
	float4 RuntimeVirtualTextureMipLevel;
	float2 RuntimeVirtualTexturePackHeight;
	float4 RuntimeVirtualTextureDebugParams;
	float OverrideLandscapeLOD;
	int FarShadowStaticMeshLODBias;
	float MinRoughness;
	float4 HairRenderInfo;
	uint EnableSkyLight;
	uint HairRenderInfoBits;
	uint HairComponents;
	float bSubsurfacePostprocessEnabled;
	float4 SSProfilesTextureSizeAndInvSize;
	float4 SSProfilesPreIntegratedTextureSizeAndInvSize;
	float3 PhysicsFieldClipmapCenter;
	float PhysicsFieldClipmapDistance;
	int PhysicsFieldClipmapResolution;
	int PhysicsFieldClipmapExponent;
	int PhysicsFieldClipmapCount;
	int PhysicsFieldTargetCount;
	int4 PhysicsFieldTargets[32];
	uint InstanceSceneDataSOAStride;
	uint GPUSceneViewId;
	uint bShadingEnergyConservation;
	uint bShadingEnergyPreservation;
	SamplerState MaterialTextureBilinearWrapedSampler;
	SamplerState MaterialTextureBilinearClampedSampler;
	Texture3D<uint4> VolumetricLightmapIndirectionTexture;
	Texture3D VolumetricLightmapBrickAmbientVector;
	Texture3D VolumetricLightmapBrickSHCoefficients0;
	Texture3D VolumetricLightmapBrickSHCoefficients1;
	Texture3D VolumetricLightmapBrickSHCoefficients2;
	Texture3D VolumetricLightmapBrickSHCoefficients3;
	Texture3D VolumetricLightmapBrickSHCoefficients4;
	Texture3D VolumetricLightmapBrickSHCoefficients5;
	Texture3D SkyBentNormalBrickTexture;
	Texture3D DirectionalLightShadowingBrickTexture;
	SamplerState VolumetricLightmapBrickAmbientVectorSampler;
	SamplerState VolumetricLightmapTextureSampler0;
	SamplerState VolumetricLightmapTextureSampler1;
	SamplerState VolumetricLightmapTextureSampler2;
	SamplerState VolumetricLightmapTextureSampler3;
	SamplerState VolumetricLightmapTextureSampler4;
	SamplerState VolumetricLightmapTextureSampler5;
	SamplerState SkyBentNormalTextureSampler;
	SamplerState DirectionalLightShadowingTextureSampler;
	Texture3D GlobalDistanceFieldPageAtlasTexture;
	Texture3D GlobalDistanceFieldCoverageAtlasTexture;
	Texture3D<uint> GlobalDistanceFieldPageTableTexture;
	Texture3D GlobalDistanceFieldMipTexture;
	Texture2D AtmosphereTransmittanceTexture;
	SamplerState AtmosphereTransmittanceTextureSampler;
	Texture2D AtmosphereIrradianceTexture;
	SamplerState AtmosphereIrradianceTextureSampler;
	Texture3D AtmosphereInscatterTexture;
	SamplerState AtmosphereInscatterTextureSampler;
	Texture2D PerlinNoiseGradientTexture;
	SamplerState PerlinNoiseGradientTextureSampler;
	Texture3D PerlinNoise3DTexture;
	SamplerState PerlinNoise3DTextureSampler;
	Texture2D<uint> SobolSamplingTexture;
	Texture2D BNDSequence_OwenScrambledSequence;
	Texture2D BNDSequence_RankingScramblingTile;
	SamplerState SharedPointWrappedSampler;
	SamplerState SharedPointClampedSampler;
	SamplerState SharedBilinearWrappedSampler;
	SamplerState SharedBilinearClampedSampler;
	SamplerState SharedBilinearAnisoClampedSampler;
	SamplerState SharedTrilinearWrappedSampler;
	SamplerState SharedTrilinearClampedSampler;
	Texture2D PreIntegratedBRDF;
	SamplerState PreIntegratedBRDFSampler;
	StructuredBuffer<float4> PrimitiveSceneData;
	StructuredBuffer<float4> InstanceSceneData;
	StructuredBuffer<float4> InstancePayloadData;
	StructuredBuffer<float4> LightmapSceneData;
	StructuredBuffer<float4> SkyIrradianceEnvironmentMap;
	Texture2D TransmittanceLutTexture;
	SamplerState TransmittanceLutTextureSampler;
	Texture2D SkyViewLutTexture;
	SamplerState SkyViewLutTextureSampler;
	Texture2D DistantSkyLightLutTexture;
	SamplerState DistantSkyLightLutTextureSampler;
	Texture3D CameraAerialPerspectiveVolume;
	SamplerState CameraAerialPerspectiveVolumeSampler;
	Texture3D HairScatteringLUTTexture;
	SamplerState HairScatteringLUTSampler;
	Texture2D LTCMatTexture;
	SamplerState LTCMatSampler;
	Texture2D LTCAmpTexture;
	SamplerState LTCAmpSampler;
	Texture2D<float2> ShadingEnergyGGXSpecTexture;
	Texture3D<float2> ShadingEnergyGGXGlassTexture;
	Texture2D<float2> ShadingEnergyClothSpecTexture;
	Texture2D<float> ShadingEnergyDiffuseTexture;
	SamplerState ShadingEnergySampler;
	Texture2D SSProfilesTexture;
	SamplerState SSProfilesSampler;
	SamplerState SSProfilesTransmissionSampler;
	Texture2DArray SSProfilesPreIntegratedTexture;
	SamplerState SSProfilesPreIntegratedSampler;
	Buffer<float4> WaterIndirection;
	Buffer<float4> WaterData;
	Buffer<uint> LandscapeIndirection;
	Buffer<float> LandscapePerComponentData;
	RWBuffer<uint> VTFeedbackBuffer;
	Buffer<uint> EditorVisualizeLevelInstanceIds;
	Buffer<uint> EditorSelectedHitProxyIds;
	Buffer<float> PhysicsFieldClipmapBuffer;
} View = {View_TranslatedWorldToClip,View_RelativeWorldToClip,View_ClipToRelativeWorld,View_TranslatedWorldToView,View_ViewToTranslatedWorld,View_TranslatedWorldToCameraView,View_CameraViewToTranslatedWorld,View_ViewToClip,View_ViewToClipNoAA,View_ClipToView,View_ClipToTranslatedWorld,View_SVPositionToTranslatedWorld,View_ScreenToRelativeWorld,View_ScreenToTranslatedWorld,View_MobileMultiviewShadowTransform,View_ViewTilePosition,View_MatrixTilePosition,View_ViewForward,View_ViewUp,View_ViewRight,View_HMDViewNoRollUp,View_HMDViewNoRollRight,View_InvDeviceZToWorldZTransform,View_ScreenPositionScaleBias,View_RelativeWorldCameraOrigin,View_TranslatedWorldCameraOrigin,View_RelativeWorldViewOrigin,View_RelativePreViewTranslation,View_PrevViewToClip,View_PrevClipToView,View_PrevTranslatedWorldToClip,View_PrevTranslatedWorldToView,View_PrevViewToTranslatedWorld,View_PrevTranslatedWorldToCameraView,View_PrevCameraViewToTranslatedWorld,View_PrevTranslatedWorldCameraOrigin,View_PrevRelativeWorldCameraOrigin,View_PrevRelativeWorldViewOrigin,View_RelativePrevPreViewTranslation,View_PrevClipToRelativeWorld,View_PrevScreenToTranslatedWorld,View_ClipToPrevClip,View_ClipToPrevClipWithAA,View_TemporalAAJitter,View_GlobalClippingPlane,View_FieldOfViewWideAngles,View_PrevFieldOfViewWideAngles,View_ViewRectMin,View_ViewSizeAndInvSize,View_LightProbeSizeRatioAndInvSizeRatio,View_BufferSizeAndInvSize,View_BufferBilinearUVMinMax,View_ScreenToViewSpace,View_NumSceneColorMSAASamples,View_PreExposure,View_OneOverPreExposure,View_DiffuseOverrideParameter,View_SpecularOverrideParameter,View_NormalOverrideParameter,View_RoughnessOverrideParameter,View_PrevFrameGameTime,View_PrevFrameRealTime,View_OutOfBoundsMask,View_WorldCameraMovementSinceLastFrame,View_CullingSign,View_NearPlane,View_GameTime,View_RealTime,View_DeltaTime,View_MaterialTextureMipBias,View_MaterialTextureDerivativeMultiply,View_Random,View_FrameNumber,View_StateFrameIndexMod8,View_StateFrameIndex,View_StateRawFrameIndex,View_AntiAliasingSampleParams,View_DebugViewModeMask,View_DebugInput0,View_CameraCut,View_UnlitViewmodeMask,View_DirectionalLightColor,View_DirectionalLightDirection,View_TranslucencyLightingVolumeMin,View_TranslucencyLightingVolumeInvSize,View_TemporalAAParams,View_CircleDOFParams,View_ForceDrawAllVelocities,View_DepthOfFieldSensorWidth,View_DepthOfFieldFocalDistance,View_DepthOfFieldScale,View_DepthOfFieldFocalLength,View_DepthOfFieldFocalRegion,View_DepthOfFieldNearTransitionRegion,View_DepthOfFieldFarTransitionRegion,View_MotionBlurNormalizedToPixel,View_GeneralPurposeTweak,View_GeneralPurposeTweak2,View_DemosaicVposOffset,View_DecalDepthBias,View_IndirectLightingColorScale,View_PrecomputedIndirectLightingColorScale,View_PrecomputedIndirectSpecularColorScale,View_AtmosphereLightDirection,View_AtmosphereLightIlluminanceOnGroundPostTransmittance,View_AtmosphereLightIlluminanceOuterSpace,View_AtmosphereLightDiscLuminance,View_AtmosphereLightDiscCosHalfApexAngle,View_SkyViewLutSizeAndInvSize,View_SkyCameraTranslatedWorldOrigin,View_SkyPlanetTranslatedWorldCenterAndViewHeight,View_SkyViewLutReferential,View_SkyAtmosphereSkyLuminanceFactor,View_SkyAtmospherePresentInScene,View_SkyAtmosphereHeightFogContribution,View_SkyAtmosphereBottomRadiusKm,View_SkyAtmosphereTopRadiusKm,View_SkyAtmosphereCameraAerialPerspectiveVolumeSizeAndInvSize,View_SkyAtmosphereAerialPerspectiveStartDepthKm,View_SkyAtmosphereCameraAerialPerspectiveVolumeDepthResolution,View_SkyAtmosphereCameraAerialPerspectiveVolumeDepthResolutionInv,View_SkyAtmosphereCameraAerialPerspectiveVolumeDepthSliceLengthKm,View_SkyAtmosphereCameraAerialPerspectiveVolumeDepthSliceLengthKmInv,View_SkyAtmosphereApplyCameraAerialPerspectiveVolume,View_NormalCurvatureToRoughnessScaleBias,View_RenderingReflectionCaptureMask,View_RealTimeReflectionCapture,View_RealTimeReflectionCapturePreExposure,View_AmbientCubemapTint,View_AmbientCubemapIntensity,View_SkyLightApplyPrecomputedBentNormalShadowingFlag,View_SkyLightAffectReflectionFlag,View_SkyLightAffectGlobalIlluminationFlag,View_SkyLightColor,View_MobileSkyIrradianceEnvironmentMap,View_MobilePreviewMode,View_HMDEyePaddingOffset,View_ReflectionCubemapMaxMip,View_ShowDecalsMask,View_DistanceFieldAOSpecularOcclusionMode,View_IndirectCapsuleSelfShadowingIntensity,View_ReflectionEnvironmentRoughnessMixingScaleBiasAndLargestWeight,View_StereoPassIndex,View_GlobalVolumeCenterAndExtent,View_GlobalVolumeWorldToUVAddAndMul,View_GlobalDistanceFieldMipWorldToUVScale,View_GlobalDistanceFieldMipWorldToUVBias,View_GlobalDistanceFieldMipFactor,View_GlobalDistanceFieldMipTransition,View_GlobalDistanceFieldClipmapSizeInPages,View_GlobalDistanceFieldInvPageAtlasSize,View_GlobalDistanceFieldInvCoverageAtlasSize,View_GlobalVolumeDimension,View_GlobalVolumeTexelSize,View_MaxGlobalDFAOConeDistance,View_NumGlobalSDFClipmaps,View_FullyCoveredExpandSurfaceScale,View_UncoveredExpandSurfaceScale,View_UncoveredMinStepScale,View_CursorPosition,View_bCheckerboardSubsurfaceProfileRendering,View_VolumetricFogInvGridSize,View_VolumetricFogGridZParams,View_VolumetricFogSVPosToVolumeUV,View_VolumetricFogMaxDistance,View_VolumetricLightmapWorldToUVScale,View_VolumetricLightmapWorldToUVAdd,View_VolumetricLightmapIndirectionTextureSize,View_VolumetricLightmapBrickSize,View_VolumetricLightmapBrickTexelSize,View_StereoIPD,View_IndirectLightingCacheShowFlag,View_EyeToPixelSpreadAngle,View_XRPassthroughCameraUVs,View_GlobalVirtualTextureMipBias,View_VirtualTextureFeedbackShift,View_VirtualTextureFeedbackMask,View_VirtualTextureFeedbackStride,View_VirtualTextureFeedbackJitterOffset,View_VirtualTextureFeedbackSampleOffset,View_RuntimeVirtualTextureMipLevel,View_RuntimeVirtualTexturePackHeight,View_RuntimeVirtualTextureDebugParams,View_OverrideLandscapeLOD,View_FarShadowStaticMeshLODBias,View_MinRoughness,View_HairRenderInfo,View_EnableSkyLight,View_HairRenderInfoBits,View_HairComponents,View_bSubsurfacePostprocessEnabled,View_SSProfilesTextureSizeAndInvSize,View_SSProfilesPreIntegratedTextureSizeAndInvSize,View_PhysicsFieldClipmapCenter,View_PhysicsFieldClipmapDistance,View_PhysicsFieldClipmapResolution,View_PhysicsFieldClipmapExponent,View_PhysicsFieldClipmapCount,View_PhysicsFieldTargetCount,View_PhysicsFieldTargets,View_InstanceSceneDataSOAStride,View_GPUSceneViewId,View_bShadingEnergyConservation,View_bShadingEnergyPreservation,View_MaterialTextureBilinearWrapedSampler,View_MaterialTextureBilinearClampedSampler,View_VolumetricLightmapIndirectionTexture,View_VolumetricLightmapBrickAmbientVector,View_VolumetricLightmapBrickSHCoefficients0,View_VolumetricLightmapBrickSHCoefficients1,View_VolumetricLightmapBrickSHCoefficients2,View_VolumetricLightmapBrickSHCoefficients3,View_VolumetricLightmapBrickSHCoefficients4,View_VolumetricLightmapBrickSHCoefficients5,View_SkyBentNormalBrickTexture,View_DirectionalLightShadowingBrickTexture,View_VolumetricLightmapBrickAmbientVectorSampler,View_VolumetricLightmapTextureSampler0,View_VolumetricLightmapTextureSampler1,View_VolumetricLightmapTextureSampler2,View_VolumetricLightmapTextureSampler3,View_VolumetricLightmapTextureSampler4,View_VolumetricLightmapTextureSampler5,View_SkyBentNormalTextureSampler,View_DirectionalLightShadowingTextureSampler,View_GlobalDistanceFieldPageAtlasTexture,View_GlobalDistanceFieldCoverageAtlasTexture,View_GlobalDistanceFieldPageTableTexture,View_GlobalDistanceFieldMipTexture,View_AtmosphereTransmittanceTexture,View_AtmosphereTransmittanceTextureSampler,View_AtmosphereIrradianceTexture,View_AtmosphereIrradianceTextureSampler,View_AtmosphereInscatterTexture,View_AtmosphereInscatterTextureSampler,View_PerlinNoiseGradientTexture,View_PerlinNoiseGradientTextureSampler,View_PerlinNoise3DTexture,View_PerlinNoise3DTextureSampler,View_SobolSamplingTexture,View_BNDSequence_OwenScrambledSequence,View_BNDSequence_RankingScramblingTile,View_SharedPointWrappedSampler,View_SharedPointClampedSampler,View_SharedBilinearWrappedSampler,View_SharedBilinearClampedSampler,View_SharedBilinearAnisoClampedSampler,View_SharedTrilinearWrappedSampler,View_SharedTrilinearClampedSampler,View_PreIntegratedBRDF,View_PreIntegratedBRDFSampler,  View_PrimitiveSceneData,   View_InstanceSceneData,   View_InstancePayloadData,   View_LightmapSceneData,   View_SkyIrradianceEnvironmentMap,  View_TransmittanceLutTexture,View_TransmittanceLutTextureSampler,View_SkyViewLutTexture,View_SkyViewLutTextureSampler,View_DistantSkyLightLutTexture,View_DistantSkyLightLutTextureSampler,View_CameraAerialPerspectiveVolume,View_CameraAerialPerspectiveVolumeSampler,View_HairScatteringLUTTexture,View_HairScatteringLUTSampler,View_LTCMatTexture,View_LTCMatSampler,View_LTCAmpTexture,View_LTCAmpSampler,View_ShadingEnergyGGXSpecTexture,View_ShadingEnergyGGXGlassTexture,View_ShadingEnergyClothSpecTexture,View_ShadingEnergyDiffuseTexture,View_ShadingEnergySampler,View_SSProfilesTexture,View_SSProfilesSampler,View_SSProfilesTransmissionSampler,View_SSProfilesPreIntegratedTexture,View_SSProfilesPreIntegratedSampler,  View_WaterIndirection,   View_WaterData,   View_LandscapeIndirection,   View_LandscapePerComponentData,  View_VTFeedbackBuffer,  View_EditorVisualizeLevelInstanceIds,   View_EditorSelectedHitProxyIds,   View_PhysicsFieldClipmapBuffer,  */
#line 4 "/Engine/Generated/GeneratedUniformBuffers.ush"
#line 1 "/Engine/Generated/UniformBuffers/Strata.ush"


cbuffer Strata
{
	uint Strata_MaxBytesPerPixel;
	uint Strata_bRoughDiffuse;
}
Texture2DArray<uint> Strata_MaterialTextureArray;
Texture2D<uint> Strata_TopLayerTexture;
Texture2D<uint2> Strata_SSSTexture;
/*atic const struct
{
	uint MaxBytesPerPixel;
	uint bRoughDiffuse;
	Texture2DArray<uint> MaterialTextureArray;
	Texture2D<uint> TopLayerTexture;
	Texture2D<uint2> SSSTexture;
} Strata = {Strata_MaxBytesPerPixel,Strata_bRoughDiffuse,Strata_MaterialTextureArray,Strata_TopLayerTexture,Strata_SSSTexture,*/
#line 5 "/Engine/Generated/GeneratedUniformBuffers.ush"
#line 1 "/Engine/Generated/UniformBuffers/DrawRectangleParameters.ush"


cbuffer DrawRectangleParameters
{
	float4 DrawRectangleParameters_PosScaleBias;
	float4 DrawRectangleParameters_UVScaleBias;
	float4 DrawRectangleParameters_InvTargetSizeAndTextureSize;
}
/*atic const struct
{
	float4 PosScaleBias;
	float4 UVScaleBias;
	float4 InvTargetSizeAndTextureSize;
} DrawRectangleParameters = {DrawRectangleParameters_PosScaleBias,DrawRectangleParameters_UVScaleBias,DrawRectangleParameters_InvTargetSizeAndTextureSize,*/
#line 6 "/Engine/Generated/GeneratedUniformBuffers.ush"
#line 1 "/Engine/Generated/UniformBuffers/InstancedView.ush"


cbuffer InstancedView
{
	float4x4 InstancedView_TranslatedWorldToClip;
	float4x4 InstancedView_RelativeWorldToClip;
	float4x4 InstancedView_ClipToRelativeWorld;
	float4x4 InstancedView_TranslatedWorldToView;
	float4x4 InstancedView_ViewToTranslatedWorld;
	float4x4 InstancedView_TranslatedWorldToCameraView;
	float4x4 InstancedView_CameraViewToTranslatedWorld;
	float4x4 InstancedView_ViewToClip;
	float4x4 InstancedView_ViewToClipNoAA;
	float4x4 InstancedView_ClipToView;
	float4x4 InstancedView_ClipToTranslatedWorld;
	float4x4 InstancedView_SVPositionToTranslatedWorld;
	float4x4 InstancedView_ScreenToRelativeWorld;
	float4x4 InstancedView_ScreenToTranslatedWorld;
	float4x4 InstancedView_MobileMultiviewShadowTransform;
	float3 InstancedView_ViewTilePosition;
	float PrePadding_InstancedView_972;
	float3 InstancedView_MatrixTilePosition;
	float PrePadding_InstancedView_988;
	float3 InstancedView_ViewForward;
	float PrePadding_InstancedView_1004;
	float3 InstancedView_ViewUp;
	float PrePadding_InstancedView_1020;
	float3 InstancedView_ViewRight;
	float PrePadding_InstancedView_1036;
	float3 InstancedView_HMDViewNoRollUp;
	float PrePadding_InstancedView_1052;
	float3 InstancedView_HMDViewNoRollRight;
	float PrePadding_InstancedView_1068;
	float4 InstancedView_InvDeviceZToWorldZTransform;
	float4 InstancedView_ScreenPositionScaleBias;
	float3 InstancedView_RelativeWorldCameraOrigin;
	float PrePadding_InstancedView_1116;
	float3 InstancedView_TranslatedWorldCameraOrigin;
	float PrePadding_InstancedView_1132;
	float3 InstancedView_RelativeWorldViewOrigin;
	float PrePadding_InstancedView_1148;
	float3 InstancedView_RelativePreViewTranslation;
	float PrePadding_InstancedView_1164;
	float4x4 InstancedView_PrevViewToClip;
	float4x4 InstancedView_PrevClipToView;
	float4x4 InstancedView_PrevTranslatedWorldToClip;
	float4x4 InstancedView_PrevTranslatedWorldToView;
	float4x4 InstancedView_PrevViewToTranslatedWorld;
	float4x4 InstancedView_PrevTranslatedWorldToCameraView;
	float4x4 InstancedView_PrevCameraViewToTranslatedWorld;
	float3 InstancedView_PrevTranslatedWorldCameraOrigin;
	float PrePadding_InstancedView_1628;
	float3 InstancedView_PrevRelativeWorldCameraOrigin;
	float PrePadding_InstancedView_1644;
	float3 InstancedView_PrevRelativeWorldViewOrigin;
	float PrePadding_InstancedView_1660;
	float3 InstancedView_RelativePrevPreViewTranslation;
	float PrePadding_InstancedView_1676;
	float4x4 InstancedView_PrevClipToRelativeWorld;
	float4x4 InstancedView_PrevScreenToTranslatedWorld;
	float4x4 InstancedView_ClipToPrevClip;
	float4x4 InstancedView_ClipToPrevClipWithAA;
	float4 InstancedView_TemporalAAJitter;
	float4 InstancedView_GlobalClippingPlane;
	float2 InstancedView_FieldOfViewWideAngles;
	float2 InstancedView_PrevFieldOfViewWideAngles;
	float4 InstancedView_ViewRectMin;
	float4 InstancedView_ViewSizeAndInvSize;
	float4 InstancedView_LightProbeSizeRatioAndInvSizeRatio;
	float4 InstancedView_BufferSizeAndInvSize;
	float4 InstancedView_BufferBilinearUVMinMax;
	float4 InstancedView_ScreenToViewSpace;
	int InstancedView_NumSceneColorMSAASamples;
	float InstancedView_PreExposure;
	float InstancedView_OneOverPreExposure;
	float PrePadding_InstancedView_2092;
	float4 InstancedView_DiffuseOverrideParameter;
	float4 InstancedView_SpecularOverrideParameter;
	float4 InstancedView_NormalOverrideParameter;
	float2 InstancedView_RoughnessOverrideParameter;
	float InstancedView_PrevFrameGameTime;
	float InstancedView_PrevFrameRealTime;
	float InstancedView_OutOfBoundsMask;
	float PrePadding_InstancedView_2164;
	float PrePadding_InstancedView_2168;
	float PrePadding_InstancedView_2172;
	float3 InstancedView_WorldCameraMovementSinceLastFrame;
	float InstancedView_CullingSign;
	float InstancedView_NearPlane;
	float InstancedView_GameTime;
	float InstancedView_RealTime;
	float InstancedView_DeltaTime;
	float InstancedView_MaterialTextureMipBias;
	float InstancedView_MaterialTextureDerivativeMultiply;
	uint InstancedView_Random;
	uint InstancedView_FrameNumber;
	uint InstancedView_StateFrameIndexMod8;
	uint InstancedView_StateFrameIndex;
	uint InstancedView_StateRawFrameIndex;
	uint PrePadding_InstancedView_2236;
	float4 InstancedView_AntiAliasingSampleParams;
	uint InstancedView_DebugViewModeMask;
	uint InstancedView_DebugInput0;
	float InstancedView_CameraCut;
	float InstancedView_UnlitViewmodeMask;
	float4 InstancedView_DirectionalLightColor;
	float3 InstancedView_DirectionalLightDirection;
	float PrePadding_InstancedView_2300;
	float4 InstancedView_TranslucencyLightingVolumeMin[2];
	float4 InstancedView_TranslucencyLightingVolumeInvSize[2];
	float4 InstancedView_TemporalAAParams;
	float4 InstancedView_CircleDOFParams;
	uint InstancedView_ForceDrawAllVelocities;
	float InstancedView_DepthOfFieldSensorWidth;
	float InstancedView_DepthOfFieldFocalDistance;
	float InstancedView_DepthOfFieldScale;
	float InstancedView_DepthOfFieldFocalLength;
	float InstancedView_DepthOfFieldFocalRegion;
	float InstancedView_DepthOfFieldNearTransitionRegion;
	float InstancedView_DepthOfFieldFarTransitionRegion;
	float InstancedView_MotionBlurNormalizedToPixel;
	float InstancedView_GeneralPurposeTweak;
	float InstancedView_GeneralPurposeTweak2;
	float InstancedView_DemosaicVposOffset;
	float InstancedView_DecalDepthBias;
	float PrePadding_InstancedView_2452;
	float PrePadding_InstancedView_2456;
	float PrePadding_InstancedView_2460;
	float3 InstancedView_IndirectLightingColorScale;
	float PrePadding_InstancedView_2476;
	float3 InstancedView_PrecomputedIndirectLightingColorScale;
	float PrePadding_InstancedView_2492;
	float3 InstancedView_PrecomputedIndirectSpecularColorScale;
	float PrePadding_InstancedView_2508;
	float4 InstancedView_AtmosphereLightDirection[2];
	float4 InstancedView_AtmosphereLightIlluminanceOnGroundPostTransmittance[2];
	float4 InstancedView_AtmosphereLightIlluminanceOuterSpace[2];
	float4 InstancedView_AtmosphereLightDiscLuminance[2];
	float4 InstancedView_AtmosphereLightDiscCosHalfApexAngle[2];
	float4 InstancedView_SkyViewLutSizeAndInvSize;
	float3 InstancedView_SkyCameraTranslatedWorldOrigin;
	float PrePadding_InstancedView_2700;
	float4 InstancedView_SkyPlanetTranslatedWorldCenterAndViewHeight;
	float4x4 InstancedView_SkyViewLutReferential;
	float4 InstancedView_SkyAtmosphereSkyLuminanceFactor;
	float InstancedView_SkyAtmospherePresentInScene;
	float InstancedView_SkyAtmosphereHeightFogContribution;
	float InstancedView_SkyAtmosphereBottomRadiusKm;
	float InstancedView_SkyAtmosphereTopRadiusKm;
	float4 InstancedView_SkyAtmosphereCameraAerialPerspectiveVolumeSizeAndInvSize;
	float InstancedView_SkyAtmosphereAerialPerspectiveStartDepthKm;
	float InstancedView_SkyAtmosphereCameraAerialPerspectiveVolumeDepthResolution;
	float InstancedView_SkyAtmosphereCameraAerialPerspectiveVolumeDepthResolutionInv;
	float InstancedView_SkyAtmosphereCameraAerialPerspectiveVolumeDepthSliceLengthKm;
	float InstancedView_SkyAtmosphereCameraAerialPerspectiveVolumeDepthSliceLengthKmInv;
	float InstancedView_SkyAtmosphereApplyCameraAerialPerspectiveVolume;
	float PrePadding_InstancedView_2856;
	float PrePadding_InstancedView_2860;
	float3 InstancedView_NormalCurvatureToRoughnessScaleBias;
	float InstancedView_RenderingReflectionCaptureMask;
	float InstancedView_RealTimeReflectionCapture;
	float InstancedView_RealTimeReflectionCapturePreExposure;
	float PrePadding_InstancedView_2888;
	float PrePadding_InstancedView_2892;
	float4 InstancedView_AmbientCubemapTint;
	float InstancedView_AmbientCubemapIntensity;
	float InstancedView_SkyLightApplyPrecomputedBentNormalShadowingFlag;
	float InstancedView_SkyLightAffectReflectionFlag;
	float InstancedView_SkyLightAffectGlobalIlluminationFlag;
	float4 InstancedView_SkyLightColor;
	float4 InstancedView_MobileSkyIrradianceEnvironmentMap[7];
	float InstancedView_MobilePreviewMode;
	float InstancedView_HMDEyePaddingOffset;
	float InstancedView_ReflectionCubemapMaxMip;
	float InstancedView_ShowDecalsMask;
	uint InstancedView_DistanceFieldAOSpecularOcclusionMode;
	float InstancedView_IndirectCapsuleSelfShadowingIntensity;
	float PrePadding_InstancedView_3080;
	float PrePadding_InstancedView_3084;
	float3 InstancedView_ReflectionEnvironmentRoughnessMixingScaleBiasAndLargestWeight;
	int InstancedView_StereoPassIndex;
	float4 InstancedView_GlobalVolumeCenterAndExtent[6];
	float4 InstancedView_GlobalVolumeWorldToUVAddAndMul[6];
	float4 InstancedView_GlobalDistanceFieldMipWorldToUVScale[6];
	float4 InstancedView_GlobalDistanceFieldMipWorldToUVBias[6];
	float InstancedView_GlobalDistanceFieldMipFactor;
	float InstancedView_GlobalDistanceFieldMipTransition;
	int InstancedView_GlobalDistanceFieldClipmapSizeInPages;
	int PrePadding_InstancedView_3500;
	float3 InstancedView_GlobalDistanceFieldInvPageAtlasSize;
	float PrePadding_InstancedView_3516;
	float3 InstancedView_GlobalDistanceFieldInvCoverageAtlasSize;
	float InstancedView_GlobalVolumeDimension;
	float InstancedView_GlobalVolumeTexelSize;
	float InstancedView_MaxGlobalDFAOConeDistance;
	uint InstancedView_NumGlobalSDFClipmaps;
	float InstancedView_FullyCoveredExpandSurfaceScale;
	float InstancedView_UncoveredExpandSurfaceScale;
	float InstancedView_UncoveredMinStepScale;
	int2 InstancedView_CursorPosition;
	float InstancedView_bCheckerboardSubsurfaceProfileRendering;
	float PrePadding_InstancedView_3572;
	float PrePadding_InstancedView_3576;
	float PrePadding_InstancedView_3580;
	float3 InstancedView_VolumetricFogInvGridSize;
	float PrePadding_InstancedView_3596;
	float3 InstancedView_VolumetricFogGridZParams;
	float PrePadding_InstancedView_3612;
	float2 InstancedView_VolumetricFogSVPosToVolumeUV;
	float InstancedView_VolumetricFogMaxDistance;
	float PrePadding_InstancedView_3628;
	float3 InstancedView_VolumetricLightmapWorldToUVScale;
	float PrePadding_InstancedView_3644;
	float3 InstancedView_VolumetricLightmapWorldToUVAdd;
	float PrePadding_InstancedView_3660;
	float3 InstancedView_VolumetricLightmapIndirectionTextureSize;
	float InstancedView_VolumetricLightmapBrickSize;
	float3 InstancedView_VolumetricLightmapBrickTexelSize;
	float InstancedView_StereoIPD;
	float InstancedView_IndirectLightingCacheShowFlag;
	float InstancedView_EyeToPixelSpreadAngle;
	float PrePadding_InstancedView_3704;
	float PrePadding_InstancedView_3708;
	float4 InstancedView_XRPassthroughCameraUVs[2];
	float InstancedView_GlobalVirtualTextureMipBias;
	uint InstancedView_VirtualTextureFeedbackShift;
	uint InstancedView_VirtualTextureFeedbackMask;
	uint InstancedView_VirtualTextureFeedbackStride;
	uint InstancedView_VirtualTextureFeedbackJitterOffset;
	uint InstancedView_VirtualTextureFeedbackSampleOffset;
	uint PrePadding_InstancedView_3768;
	uint PrePadding_InstancedView_3772;
	float4 InstancedView_RuntimeVirtualTextureMipLevel;
	float2 InstancedView_RuntimeVirtualTexturePackHeight;
	float PrePadding_InstancedView_3800;
	float PrePadding_InstancedView_3804;
	float4 InstancedView_RuntimeVirtualTextureDebugParams;
	float InstancedView_OverrideLandscapeLOD;
	int InstancedView_FarShadowStaticMeshLODBias;
	float InstancedView_MinRoughness;
	float PrePadding_InstancedView_3836;
	float4 InstancedView_HairRenderInfo;
	uint InstancedView_EnableSkyLight;
	uint InstancedView_HairRenderInfoBits;
	uint InstancedView_HairComponents;
	float InstancedView_bSubsurfacePostprocessEnabled;
	float4 InstancedView_SSProfilesTextureSizeAndInvSize;
	float4 InstancedView_SSProfilesPreIntegratedTextureSizeAndInvSize;
	float3 InstancedView_PhysicsFieldClipmapCenter;
	float InstancedView_PhysicsFieldClipmapDistance;
	int InstancedView_PhysicsFieldClipmapResolution;
	int InstancedView_PhysicsFieldClipmapExponent;
	int InstancedView_PhysicsFieldClipmapCount;
	int InstancedView_PhysicsFieldTargetCount;
	int4 InstancedView_PhysicsFieldTargets[32];
	uint InstancedView_InstanceSceneDataSOAStride;
	uint InstancedView_GPUSceneViewId;
}
/*atic const struct
{
	float4x4 TranslatedWorldToClip;
	float4x4 RelativeWorldToClip;
	float4x4 ClipToRelativeWorld;
	float4x4 TranslatedWorldToView;
	float4x4 ViewToTranslatedWorld;
	float4x4 TranslatedWorldToCameraView;
	float4x4 CameraViewToTranslatedWorld;
	float4x4 ViewToClip;
	float4x4 ViewToClipNoAA;
	float4x4 ClipToView;
	float4x4 ClipToTranslatedWorld;
	float4x4 SVPositionToTranslatedWorld;
	float4x4 ScreenToRelativeWorld;
	float4x4 ScreenToTranslatedWorld;
	float4x4 MobileMultiviewShadowTransform;
	float3 ViewTilePosition;
	float3 MatrixTilePosition;
	float3 ViewForward;
	float3 ViewUp;
	float3 ViewRight;
	float3 HMDViewNoRollUp;
	float3 HMDViewNoRollRight;
	float4 InvDeviceZToWorldZTransform;
	float4 ScreenPositionScaleBias;
	float3 RelativeWorldCameraOrigin;
	float3 TranslatedWorldCameraOrigin;
	float3 RelativeWorldViewOrigin;
	float3 RelativePreViewTranslation;
	float4x4 PrevViewToClip;
	float4x4 PrevClipToView;
	float4x4 PrevTranslatedWorldToClip;
	float4x4 PrevTranslatedWorldToView;
	float4x4 PrevViewToTranslatedWorld;
	float4x4 PrevTranslatedWorldToCameraView;
	float4x4 PrevCameraViewToTranslatedWorld;
	float3 PrevTranslatedWorldCameraOrigin;
	float3 PrevRelativeWorldCameraOrigin;
	float3 PrevRelativeWorldViewOrigin;
	float3 RelativePrevPreViewTranslation;
	float4x4 PrevClipToRelativeWorld;
	float4x4 PrevScreenToTranslatedWorld;
	float4x4 ClipToPrevClip;
	float4x4 ClipToPrevClipWithAA;
	float4 TemporalAAJitter;
	float4 GlobalClippingPlane;
	float2 FieldOfViewWideAngles;
	float2 PrevFieldOfViewWideAngles;
	float4 ViewRectMin;
	float4 ViewSizeAndInvSize;
	float4 LightProbeSizeRatioAndInvSizeRatio;
	float4 BufferSizeAndInvSize;
	float4 BufferBilinearUVMinMax;
	float4 ScreenToViewSpace;
	int NumSceneColorMSAASamples;
	float PreExposure;
	float OneOverPreExposure;
	float4 DiffuseOverrideParameter;
	float4 SpecularOverrideParameter;
	float4 NormalOverrideParameter;
	float2 RoughnessOverrideParameter;
	float PrevFrameGameTime;
	float PrevFrameRealTime;
	float OutOfBoundsMask;
	float3 WorldCameraMovementSinceLastFrame;
	float CullingSign;
	float NearPlane;
	float GameTime;
	float RealTime;
	float DeltaTime;
	float MaterialTextureMipBias;
	float MaterialTextureDerivativeMultiply;
	uint Random;
	uint FrameNumber;
	uint StateFrameIndexMod8;
	uint StateFrameIndex;
	uint StateRawFrameIndex;
	float4 AntiAliasingSampleParams;
	uint DebugViewModeMask;
	uint DebugInput0;
	float CameraCut;
	float UnlitViewmodeMask;
	float4 DirectionalLightColor;
	float3 DirectionalLightDirection;
	float4 TranslucencyLightingVolumeMin[2];
	float4 TranslucencyLightingVolumeInvSize[2];
	float4 TemporalAAParams;
	float4 CircleDOFParams;
	uint ForceDrawAllVelocities;
	float DepthOfFieldSensorWidth;
	float DepthOfFieldFocalDistance;
	float DepthOfFieldScale;
	float DepthOfFieldFocalLength;
	float DepthOfFieldFocalRegion;
	float DepthOfFieldNearTransitionRegion;
	float DepthOfFieldFarTransitionRegion;
	float MotionBlurNormalizedToPixel;
	float GeneralPurposeTweak;
	float GeneralPurposeTweak2;
	float DemosaicVposOffset;
	float DecalDepthBias;
	float3 IndirectLightingColorScale;
	float3 PrecomputedIndirectLightingColorScale;
	float3 PrecomputedIndirectSpecularColorScale;
	float4 AtmosphereLightDirection[2];
	float4 AtmosphereLightIlluminanceOnGroundPostTransmittance[2];
	float4 AtmosphereLightIlluminanceOuterSpace[2];
	float4 AtmosphereLightDiscLuminance[2];
	float4 AtmosphereLightDiscCosHalfApexAngle[2];
	float4 SkyViewLutSizeAndInvSize;
	float3 SkyCameraTranslatedWorldOrigin;
	float4 SkyPlanetTranslatedWorldCenterAndViewHeight;
	float4x4 SkyViewLutReferential;
	float4 SkyAtmosphereSkyLuminanceFactor;
	float SkyAtmospherePresentInScene;
	float SkyAtmosphereHeightFogContribution;
	float SkyAtmosphereBottomRadiusKm;
	float SkyAtmosphereTopRadiusKm;
	float4 SkyAtmosphereCameraAerialPerspectiveVolumeSizeAndInvSize;
	float SkyAtmosphereAerialPerspectiveStartDepthKm;
	float SkyAtmosphereCameraAerialPerspectiveVolumeDepthResolution;
	float SkyAtmosphereCameraAerialPerspectiveVolumeDepthResolutionInv;
	float SkyAtmosphereCameraAerialPerspectiveVolumeDepthSliceLengthKm;
	float SkyAtmosphereCameraAerialPerspectiveVolumeDepthSliceLengthKmInv;
	float SkyAtmosphereApplyCameraAerialPerspectiveVolume;
	float3 NormalCurvatureToRoughnessScaleBias;
	float RenderingReflectionCaptureMask;
	float RealTimeReflectionCapture;
	float RealTimeReflectionCapturePreExposure;
	float4 AmbientCubemapTint;
	float AmbientCubemapIntensity;
	float SkyLightApplyPrecomputedBentNormalShadowingFlag;
	float SkyLightAffectReflectionFlag;
	float SkyLightAffectGlobalIlluminationFlag;
	float4 SkyLightColor;
	float4 MobileSkyIrradianceEnvironmentMap[7];
	float MobilePreviewMode;
	float HMDEyePaddingOffset;
	float ReflectionCubemapMaxMip;
	float ShowDecalsMask;
	uint DistanceFieldAOSpecularOcclusionMode;
	float IndirectCapsuleSelfShadowingIntensity;
	float3 ReflectionEnvironmentRoughnessMixingScaleBiasAndLargestWeight;
	int StereoPassIndex;
	float4 GlobalVolumeCenterAndExtent[6];
	float4 GlobalVolumeWorldToUVAddAndMul[6];
	float4 GlobalDistanceFieldMipWorldToUVScale[6];
	float4 GlobalDistanceFieldMipWorldToUVBias[6];
	float GlobalDistanceFieldMipFactor;
	float GlobalDistanceFieldMipTransition;
	int GlobalDistanceFieldClipmapSizeInPages;
	float3 GlobalDistanceFieldInvPageAtlasSize;
	float3 GlobalDistanceFieldInvCoverageAtlasSize;
	float GlobalVolumeDimension;
	float GlobalVolumeTexelSize;
	float MaxGlobalDFAOConeDistance;
	uint NumGlobalSDFClipmaps;
	float FullyCoveredExpandSurfaceScale;
	float UncoveredExpandSurfaceScale;
	float UncoveredMinStepScale;
	int2 CursorPosition;
	float bCheckerboardSubsurfaceProfileRendering;
	float3 VolumetricFogInvGridSize;
	float3 VolumetricFogGridZParams;
	float2 VolumetricFogSVPosToVolumeUV;
	float VolumetricFogMaxDistance;
	float3 VolumetricLightmapWorldToUVScale;
	float3 VolumetricLightmapWorldToUVAdd;
	float3 VolumetricLightmapIndirectionTextureSize;
	float VolumetricLightmapBrickSize;
	float3 VolumetricLightmapBrickTexelSize;
	float StereoIPD;
	float IndirectLightingCacheShowFlag;
	float EyeToPixelSpreadAngle;
	float4 XRPassthroughCameraUVs[2];
	float GlobalVirtualTextureMipBias;
	uint VirtualTextureFeedbackShift;
	uint VirtualTextureFeedbackMask;
	uint VirtualTextureFeedbackStride;
	uint VirtualTextureFeedbackJitterOffset;
	uint VirtualTextureFeedbackSampleOffset;
	float4 RuntimeVirtualTextureMipLevel;
	float2 RuntimeVirtualTexturePackHeight;
	float4 RuntimeVirtualTextureDebugParams;
	float OverrideLandscapeLOD;
	int FarShadowStaticMeshLODBias;
	float MinRoughness;
	float4 HairRenderInfo;
	uint EnableSkyLight;
	uint HairRenderInfoBits;
	uint HairComponents;
	float bSubsurfacePostprocessEnabled;
	float4 SSProfilesTextureSizeAndInvSize;
	float4 SSProfilesPreIntegratedTextureSizeAndInvSize;
	float3 PhysicsFieldClipmapCenter;
	float PhysicsFieldClipmapDistance;
	int PhysicsFieldClipmapResolution;
	int PhysicsFieldClipmapExponent;
	int PhysicsFieldClipmapCount;
	int PhysicsFieldTargetCount;
	int4 PhysicsFieldTargets[32];
	uint InstanceSceneDataSOAStride;
	uint GPUSceneViewId;
} InstancedView = {InstancedView_TranslatedWorldToClip,InstancedView_RelativeWorldToClip,InstancedView_ClipToRelativeWorld,InstancedView_TranslatedWorldToView,InstancedView_ViewToTranslatedWorld,InstancedView_TranslatedWorldToCameraView,InstancedView_CameraViewToTranslatedWorld,InstancedView_ViewToClip,InstancedView_ViewToClipNoAA,InstancedView_ClipToView,InstancedView_ClipToTranslatedWorld,InstancedView_SVPositionToTranslatedWorld,InstancedView_ScreenToRelativeWorld,InstancedView_ScreenToTranslatedWorld,InstancedView_MobileMultiviewShadowTransform,InstancedView_ViewTilePosition,InstancedView_MatrixTilePosition,InstancedView_ViewForward,InstancedView_ViewUp,InstancedView_ViewRight,InstancedView_HMDViewNoRollUp,InstancedView_HMDViewNoRollRight,InstancedView_InvDeviceZToWorldZTransform,InstancedView_ScreenPositionScaleBias,InstancedView_RelativeWorldCameraOrigin,InstancedView_TranslatedWorldCameraOrigin,InstancedView_RelativeWorldViewOrigin,InstancedView_RelativePreViewTranslation,InstancedView_PrevViewToClip,InstancedView_PrevClipToView,InstancedView_PrevTranslatedWorldToClip,InstancedView_PrevTranslatedWorldToView,InstancedView_PrevViewToTranslatedWorld,InstancedView_PrevTranslatedWorldToCameraView,InstancedView_PrevCameraViewToTranslatedWorld,InstancedView_PrevTranslatedWorldCameraOrigin,InstancedView_PrevRelativeWorldCameraOrigin,InstancedView_PrevRelativeWorldViewOrigin,InstancedView_RelativePrevPreViewTranslation,InstancedView_PrevClipToRelativeWorld,InstancedView_PrevScreenToTranslatedWorld,InstancedView_ClipToPrevClip,InstancedView_ClipToPrevClipWithAA,InstancedView_TemporalAAJitter,InstancedView_GlobalClippingPlane,InstancedView_FieldOfViewWideAngles,InstancedView_PrevFieldOfViewWideAngles,InstancedView_ViewRectMin,InstancedView_ViewSizeAndInvSize,InstancedView_LightProbeSizeRatioAndInvSizeRatio,InstancedView_BufferSizeAndInvSize,InstancedView_BufferBilinearUVMinMax,InstancedView_ScreenToViewSpace,InstancedView_NumSceneColorMSAASamples,InstancedView_PreExposure,InstancedView_OneOverPreExposure,InstancedView_DiffuseOverrideParameter,InstancedView_SpecularOverrideParameter,InstancedView_NormalOverrideParameter,InstancedView_RoughnessOverrideParameter,InstancedView_PrevFrameGameTime,InstancedView_PrevFrameRealTime,InstancedView_OutOfBoundsMask,InstancedView_WorldCameraMovementSinceLastFrame,InstancedView_CullingSign,InstancedView_NearPlane,InstancedView_GameTime,InstancedView_RealTime,InstancedView_DeltaTime,InstancedView_MaterialTextureMipBias,InstancedView_MaterialTextureDerivativeMultiply,InstancedView_Random,InstancedView_FrameNumber,InstancedView_StateFrameIndexMod8,InstancedView_StateFrameIndex,InstancedView_StateRawFrameIndex,InstancedView_AntiAliasingSampleParams,InstancedView_DebugViewModeMask,InstancedView_DebugInput0,InstancedView_CameraCut,InstancedView_UnlitViewmodeMask,InstancedView_DirectionalLightColor,InstancedView_DirectionalLightDirection,InstancedView_TranslucencyLightingVolumeMin,InstancedView_TranslucencyLightingVolumeInvSize,InstancedView_TemporalAAParams,InstancedView_CircleDOFParams,InstancedView_ForceDrawAllVelocities,InstancedView_DepthOfFieldSensorWidth,InstancedView_DepthOfFieldFocalDistance,InstancedView_DepthOfFieldScale,InstancedView_DepthOfFieldFocalLength,InstancedView_DepthOfFieldFocalRegion,InstancedView_DepthOfFieldNearTransitionRegion,InstancedView_DepthOfFieldFarTransitionRegion,InstancedView_MotionBlurNormalizedToPixel,InstancedView_GeneralPurposeTweak,InstancedView_GeneralPurposeTweak2,InstancedView_DemosaicVposOffset,InstancedView_DecalDepthBias,InstancedView_IndirectLightingColorScale,InstancedView_PrecomputedIndirectLightingColorScale,InstancedView_PrecomputedIndirectSpecularColorScale,InstancedView_AtmosphereLightDirection,InstancedView_AtmosphereLightIlluminanceOnGroundPostTransmittance,InstancedView_AtmosphereLightIlluminanceOuterSpace,InstancedView_AtmosphereLightDiscLuminance,InstancedView_AtmosphereLightDiscCosHalfApexAngle,InstancedView_SkyViewLutSizeAndInvSize,InstancedView_SkyCameraTranslatedWorldOrigin,InstancedView_SkyPlanetTranslatedWorldCenterAndViewHeight,InstancedView_SkyViewLutReferential,InstancedView_SkyAtmosphereSkyLuminanceFactor,InstancedView_SkyAtmospherePresentInScene,InstancedView_SkyAtmosphereHeightFogContribution,InstancedView_SkyAtmosphereBottomRadiusKm,InstancedView_SkyAtmosphereTopRadiusKm,InstancedView_SkyAtmosphereCameraAerialPerspectiveVolumeSizeAndInvSize,InstancedView_SkyAtmosphereAerialPerspectiveStartDepthKm,InstancedView_SkyAtmosphereCameraAerialPerspectiveVolumeDepthResolution,InstancedView_SkyAtmosphereCameraAerialPerspectiveVolumeDepthResolutionInv,InstancedView_SkyAtmosphereCameraAerialPerspectiveVolumeDepthSliceLengthKm,InstancedView_SkyAtmosphereCameraAerialPerspectiveVolumeDepthSliceLengthKmInv,InstancedView_SkyAtmosphereApplyCameraAerialPerspectiveVolume,InstancedView_NormalCurvatureToRoughnessScaleBias,InstancedView_RenderingReflectionCaptureMask,InstancedView_RealTimeReflectionCapture,InstancedView_RealTimeReflectionCapturePreExposure,InstancedView_AmbientCubemapTint,InstancedView_AmbientCubemapIntensity,InstancedView_SkyLightApplyPrecomputedBentNormalShadowingFlag,InstancedView_SkyLightAffectReflectionFlag,InstancedView_SkyLightAffectGlobalIlluminationFlag,InstancedView_SkyLightColor,InstancedView_MobileSkyIrradianceEnvironmentMap,InstancedView_MobilePreviewMode,InstancedView_HMDEyePaddingOffset,InstancedView_ReflectionCubemapMaxMip,InstancedView_ShowDecalsMask,InstancedView_DistanceFieldAOSpecularOcclusionMode,InstancedView_IndirectCapsuleSelfShadowingIntensity,InstancedView_ReflectionEnvironmentRoughnessMixingScaleBiasAndLargestWeight,InstancedView_StereoPassIndex,InstancedView_GlobalVolumeCenterAndExtent,InstancedView_GlobalVolumeWorldToUVAddAndMul,InstancedView_GlobalDistanceFieldMipWorldToUVScale,InstancedView_GlobalDistanceFieldMipWorldToUVBias,InstancedView_GlobalDistanceFieldMipFactor,InstancedView_GlobalDistanceFieldMipTransition,InstancedView_GlobalDistanceFieldClipmapSizeInPages,InstancedView_GlobalDistanceFieldInvPageAtlasSize,InstancedView_GlobalDistanceFieldInvCoverageAtlasSize,InstancedView_GlobalVolumeDimension,InstancedView_GlobalVolumeTexelSize,InstancedView_MaxGlobalDFAOConeDistance,InstancedView_NumGlobalSDFClipmaps,InstancedView_FullyCoveredExpandSurfaceScale,InstancedView_UncoveredExpandSurfaceScale,InstancedView_UncoveredMinStepScale,InstancedView_CursorPosition,InstancedView_bCheckerboardSubsurfaceProfileRendering,InstancedView_VolumetricFogInvGridSize,InstancedView_VolumetricFogGridZParams,InstancedView_VolumetricFogSVPosToVolumeUV,InstancedView_VolumetricFogMaxDistance,InstancedView_VolumetricLightmapWorldToUVScale,InstancedView_VolumetricLightmapWorldToUVAdd,InstancedView_VolumetricLightmapIndirectionTextureSize,InstancedView_VolumetricLightmapBrickSize,InstancedView_VolumetricLightmapBrickTexelSize,InstancedView_StereoIPD,InstancedView_IndirectLightingCacheShowFlag,InstancedView_EyeToPixelSpreadAngle,InstancedView_XRPassthroughCameraUVs,InstancedView_GlobalVirtualTextureMipBias,InstancedView_VirtualTextureFeedbackShift,InstancedView_VirtualTextureFeedbackMask,InstancedView_VirtualTextureFeedbackStride,InstancedView_VirtualTextureFeedbackJitterOffset,InstancedView_VirtualTextureFeedbackSampleOffset,InstancedView_RuntimeVirtualTextureMipLevel,InstancedView_RuntimeVirtualTexturePackHeight,InstancedView_RuntimeVirtualTextureDebugParams,InstancedView_OverrideLandscapeLOD,InstancedView_FarShadowStaticMeshLODBias,InstancedView_MinRoughness,InstancedView_HairRenderInfo,InstancedView_EnableSkyLight,InstancedView_HairRenderInfoBits,InstancedView_HairComponents,InstancedView_bSubsurfacePostprocessEnabled,InstancedView_SSProfilesTextureSizeAndInvSize,InstancedView_SSProfilesPreIntegratedTextureSizeAndInvSize,InstancedView_PhysicsFieldClipmapCenter,InstancedView_PhysicsFieldClipmapDistance,InstancedView_PhysicsFieldClipmapResolution,InstancedView_PhysicsFieldClipmapExponent,InstancedView_PhysicsFieldClipmapCount,InstancedView_PhysicsFieldTargetCount,InstancedView_PhysicsFieldTargets,InstancedView_InstanceSceneDataSOAStride,InstancedView_GPUSceneViewId,*/
#line 7 "/Engine/Generated/GeneratedUniformBuffers.ush"
#line 170 "/Engine/Private/Common.ush"
#line 172 "/Engine/Private/Common.ush"
#line 1 "CommonViewUniformBuffer.ush"
#line 12 "/Engine/Private/CommonViewUniformBuffer.ush"
float2 GetTanHalfFieldOfView()
{
	return float2(View_ClipToView[0][0], View_ClipToView[1][1]);
}

float2 GetPrevTanHalfFieldOfView()
{
	return float2(View_PrevClipToView[0][0], View_PrevClipToView[1][1]);
}



float2 GetCotanHalfFieldOfView()
{
	return float2(View_ViewToClip[0][0], View_ViewToClip[1][1]);
}



float2 GetPrevCotanHalfFieldOfView()
{
	return float2(View_PrevViewToClip[0][0], View_PrevViewToClip[1][1]);
}


uint GetPowerOfTwoModulatedFrameIndex(uint Pow2Modulus)
{

	return View_StateFrameIndex & uint(Pow2Modulus - 1);
}
#line 173 "/Engine/Private/Common.ush"



float FmodFloor(float Lhs, float Rhs)
{
	return Lhs - floor(Lhs / Rhs) * Rhs;
}

float2 FmodFloor(float2 Lhs, float2 Rhs)
{
	return Lhs - floor(Lhs / Rhs) * Rhs;
}

float3 FmodFloor(float3 Lhs, float3 Rhs)
{
	return Lhs - floor(Lhs / Rhs) * Rhs;
}

float4 FmodFloor(float4 Lhs, float4 Rhs)
{
	return Lhs - floor(Lhs / Rhs) * Rhs;
}
#line 196 "/Engine/Private/Common.ush"
#line 1 "LargeWorldCoordinates.ush"
#line 5 "/Engine/Private/LargeWorldCoordinates.ush"
struct FLWCScalar
{
	float Tile;
	float Offset;
};

struct FLWCVector2
{
	float2 Tile;
	float2 Offset;
};

struct FLWCVector3
{
	float3 Tile;
	float3 Offset;
};

struct FLWCVector4
{
	float4 Tile;
	float4 Offset;
};


struct FLWCMatrix
{
	float4x4 M;
	float3 Tile;
};


struct FLWCInverseMatrix
{
	float4x4 M;
	float3 Tile;
	int Dummy;
};




float LWCGetTileOffset(FLWCScalar V) { return  ((V).Tile)  *  2097152.00f ; }
float2 LWCGetTileOffset(FLWCVector2 V) { return  ((V).Tile)  *  2097152.00f ; }
float3 LWCGetTileOffset(FLWCVector3 V) { return  ((V).Tile)  *  2097152.00f ; }
float4 LWCGetTileOffset(FLWCVector4 V) { return  ((V).Tile)  *  2097152.00f ; }
float3 LWCGetTileOffset(FLWCMatrix V) { return  ((V).Tile)  *  2097152.00f ; }
float3 LWCGetTileOffset(FLWCInverseMatrix V) { return  ((V).Tile)  *  2097152.00f ; }

float4x4 Make4x3Matrix(float4x4 M)
{

	float4x4 Result;
	Result[0] = float4(M[0].xyz, 0.0f);
	Result[1] = float4(M[1].xyz, 0.0f);
	Result[2] = float4(M[2].xyz, 0.0f);
	Result[3] = float4(M[3].xyz, 1.0f);
	return Result;
}

float4x4 MakeTranslationMatrix(float3 Offset)
{
	float4x4 Result;
	Result[0] = float4(1.0f, 0.0f, 0.0f, 0.0f);
	Result[1] = float4(0.0f, 1.0f, 0.0f, 0.0f);
	Result[2] = float4(0.0f, 0.0f, 1.0f, 0.0f);
	Result[3] = float4(Offset, 1.0f);
	return Result;
}

FLWCScalar MakeLWCScalar(float Tile, float Offset)
{
	FLWCScalar Result;
	(Result).Tile = (Tile) ;
	Result.Offset = Offset;
	return Result;
}

FLWCVector2 MakeLWCVector2(float2 Tile, float2 Offset)
{
	FLWCVector2 Result;
	(Result).Tile = (Tile) ;
	Result.Offset = Offset;
	return Result;
}

FLWCVector3 MakeLWCVector3(float3 Tile, float3 Offset)
{
	FLWCVector3 Result;
	(Result).Tile = (Tile) ;
	Result.Offset = Offset;
	return Result;
}

FLWCVector4 MakeLWCVector4(float4 Tile, float4 Offset)
{
	FLWCVector4 Result;
	(Result).Tile = (Tile) ;
	Result.Offset = Offset;
	return Result;
}

FLWCVector4 MakeLWCVector4(float3 Tile, float4 Offset)
{
	return MakeLWCVector4(float4(Tile, 0), Offset);
}

FLWCVector4 MakeLWCVector4(FLWCVector3 XYZ, float W)
{
	return MakeLWCVector4( ((XYZ).Tile) , float4(XYZ.Offset, W));
}

FLWCScalar MakeLWCVector(FLWCScalar X) { return X; }

FLWCVector2 MakeLWCVector(FLWCScalar X, FLWCScalar Y) { return MakeLWCVector2(float2( ((X).Tile) ,  ((Y).Tile) ), float2(X.Offset, Y.Offset)); }

FLWCVector3 MakeLWCVector(FLWCScalar X, FLWCScalar Y, FLWCScalar Z) { return MakeLWCVector3(float3( ((X).Tile) ,  ((Y).Tile) ,  ((Z).Tile) ), float3(X.Offset, Y.Offset, Z.Offset)); }
FLWCVector3 MakeLWCVector(FLWCScalar X, FLWCVector2 YZ) { return MakeLWCVector3(float3( ((X).Tile) ,  ((YZ).Tile) ), float3(X.Offset, YZ.Offset)); }
FLWCVector3 MakeLWCVector(FLWCVector2 XY, FLWCScalar Z) { return MakeLWCVector3(float3( ((XY).Tile) ,  ((Z).Tile) ), float3(XY.Offset, Z.Offset)); }

FLWCVector4 MakeLWCVector(FLWCScalar X, FLWCScalar Y, FLWCScalar Z, FLWCScalar W) { return MakeLWCVector4(float4( ((X).Tile) ,  ((Y).Tile) ,  ((Z).Tile) ,  ((W).Tile) ), float4(X.Offset, Y.Offset, Z.Offset, W.Offset)); }
FLWCVector4 MakeLWCVector(FLWCScalar X, FLWCScalar Y, FLWCVector2 ZW) { return MakeLWCVector4(float4( ((X).Tile) ,  ((Y).Tile) ,  ((ZW).Tile) ), float4(X.Offset, Y.Offset, ZW.Offset)); }
FLWCVector4 MakeLWCVector(FLWCScalar X, FLWCVector2 YZ, FLWCScalar W) { return MakeLWCVector4(float4( ((X).Tile) ,  ((YZ).Tile) ,  ((W).Tile) ), float4(X.Offset, YZ.Offset, W.Offset)); }
FLWCVector4 MakeLWCVector(FLWCVector2 XY, FLWCScalar Z, FLWCScalar W) { return MakeLWCVector4(float4( ((XY).Tile) ,  ((Z).Tile) ,  ((W).Tile) ), float4(XY.Offset, Z.Offset, W.Offset)); }
FLWCVector4 MakeLWCVector(FLWCVector2 XY, FLWCVector2 ZW) { return MakeLWCVector4(float4( ((XY).Tile) ,  ((ZW).Tile) ), float4(XY.Offset, ZW.Offset)); }
FLWCVector4 MakeLWCVector(FLWCScalar X, FLWCVector3 YZW) { return MakeLWCVector4(float4( ((X).Tile) ,  ((YZW).Tile) ), float4(X.Offset, YZW.Offset)); }
FLWCVector4 MakeLWCVector(FLWCVector3 XYZ, FLWCScalar W) { return MakeLWCVector4(float4( ((XYZ).Tile) ,  ((W).Tile) ), float4(XYZ.Offset, W.Offset)); }

FLWCMatrix MakeLWCMatrix(float3 Tile, float4x4 InMatrix)
{
	FLWCMatrix Result;
	(Result).Tile = (Tile) ;
	Result.M = InMatrix;
	return Result;
}

FLWCMatrix MakeLWCMatrix4x3(float3 Tile, float4x4 InMatrix)
{
	FLWCMatrix Result;
	(Result).Tile = (Tile) ;
	Result.M = Make4x3Matrix(InMatrix);
	return Result;
}

FLWCInverseMatrix MakeLWCInverseMatrix(float3 Tile, float4x4 InMatrix)
{
	FLWCInverseMatrix Result;
	(Result).Tile = (-Tile) ;
	Result.M = InMatrix;
	Result.Dummy = 0;
	return Result;
}

FLWCInverseMatrix MakeLWCInverseMatrix4x3(float3 Tile, float4x4 InMatrix)
{
	FLWCInverseMatrix Result;
	(Result).Tile = (-Tile) ;
	Result.M = Make4x3Matrix(InMatrix);
	Result.Dummy = 0;
	return Result;
}



FLWCScalar LWCGetComponent(FLWCScalar V, int C) { return V; }
FLWCScalar LWCGetComponent(FLWCVector2 V, int C) { return MakeLWCScalar( ((V).Tile) [C], V.Offset[C]); }
FLWCScalar LWCGetComponent(FLWCVector3 V, int C) { return MakeLWCScalar( ((V).Tile) [C], V.Offset[C]); }
FLWCScalar LWCGetComponent(FLWCVector4 V, int C) { return MakeLWCScalar( ((V).Tile) [C], V.Offset[C]); }






float LWCToFloat(FLWCScalar Value) { return LWCGetTileOffset(Value) + Value.Offset; }
float2 LWCToFloat(FLWCVector2 Value) { return LWCGetTileOffset(Value) + Value.Offset; }
float3 LWCToFloat(FLWCVector3 Value) { return LWCGetTileOffset(Value) + Value.Offset; }
float4 LWCToFloat(FLWCVector4 Value) { return LWCGetTileOffset(Value) + Value.Offset; }

float4x4 LWCToFloat(FLWCMatrix Value)
{
	float4x4 Result = Value.M;
	Result[3].xyz = LWCGetTileOffset(Value) + Result[3].xyz;
	return Result;
}

float4x4 LWCToFloat(FLWCInverseMatrix Value)
{
	float4x4 TileOffset = MakeTranslationMatrix(LWCGetTileOffset(Value));
	return mul(TileOffset, Value.M);
}

float3x3 LWCToFloat3x3(FLWCMatrix Value)
{
	return (float3x3)Value.M;
}

float3x3 LWCToFloat3x3(FLWCInverseMatrix Value)
{
	return (float3x3)Value.M;
}


float LWCToFloat(float Value) { return Value; }
float2 LWCToFloat(float2 Value) { return Value; }
float3 LWCToFloat(float3 Value) { return Value; }
float4 LWCToFloat(float4 Value) { return Value; }
float4x4 LWCToFloat(float4x4 Value) { return Value; }


FLWCScalar LWCPromote(FLWCScalar Value) { return Value; }
FLWCVector2 LWCPromote(FLWCVector2 Value) { return Value; }
FLWCVector3 LWCPromote(FLWCVector3 Value) { return Value; }
FLWCVector4 LWCPromote(FLWCVector4 Value) { return Value; }
FLWCMatrix LWCPromote(FLWCMatrix Value) { return Value; }
FLWCInverseMatrix LWCPromote(FLWCInverseMatrix Value) { return Value; }

FLWCScalar LWCPromote(float Value) { return MakeLWCScalar(0, Value); }
FLWCVector2 LWCPromote(float2 Value) { return MakeLWCVector2((float2)0, Value); }
FLWCVector3 LWCPromote(float3 Value) { return MakeLWCVector3((float3)0, Value); }
FLWCVector4 LWCPromote(float4 Value) { return MakeLWCVector4((float4)0, Value); }
FLWCMatrix LWCPromote(float4x4 Value) { return MakeLWCMatrix((float3)0, Value); }
FLWCInverseMatrix LWCPromoteInverse(float4x4 Value) { return MakeLWCInverseMatrix((float3)0, Value); }

FLWCVector3 LWCMultiply(float3 Position, FLWCMatrix InMatrix)
{

	float3 Offset = (Position.xxx * InMatrix.M[0].xyz + Position.yyy * InMatrix.M[1].xyz + Position.zzz * InMatrix.M[2].xyz) + InMatrix.M[3].xyz;
	return MakeLWCVector3( ((InMatrix).Tile) , Offset);
}

FLWCVector3 LWCInvariantMultiply(float3 Position, FLWCMatrix InMatrix)
{

	float3 Offset =  (Position.xxx * InMatrix.M[0].xyz + Position.yyy * InMatrix.M[1].xyz + Position.zzz * InMatrix.M[2].xyz) + InMatrix.M[3].xyz ;
	return MakeLWCVector3( ((InMatrix).Tile) , Offset);
}

FLWCVector4 LWCMultiply(float4 Position, FLWCMatrix InMatrix)
{
	float4 Offset = mul(Position, InMatrix.M);
	return MakeLWCVector4( ((InMatrix).Tile) , Offset);
}

float3 LWCMultiply(FLWCVector3 Position, FLWCInverseMatrix InMatrix)
{
	float3 LocalPosition = LWCToFloat(MakeLWCVector3( ((Position).Tile)  +  ((InMatrix).Tile) , Position.Offset));
	return (LocalPosition.xxx * InMatrix.M[0].xyz + LocalPosition.yyy * InMatrix.M[1].xyz + LocalPosition.zzz * InMatrix.M[2].xyz) + InMatrix.M[3].xyz;
}

float4 LWCMultiply(FLWCVector4 Position, FLWCInverseMatrix InMatrix)
{
	float4 LocalPosition = LWCToFloat(MakeLWCVector4( ((Position).Tile)  + float4( ((InMatrix).Tile) , 0), Position.Offset));
	return mul(LocalPosition, InMatrix.M);
}

float3 LWCMultiplyVector(float3 Vector, FLWCMatrix InMatrix)
{
	return mul(Vector, (float3x3)InMatrix.M);
}

float3 LWCMultiplyVector(float3 Vector, FLWCInverseMatrix InMatrix)
{
	return mul(Vector, (float3x3)InMatrix.M);
}

FLWCMatrix LWCMultiply(float4x4 Lhs, FLWCMatrix Rhs)
{
	float4x4 ResultMatrix = mul(Lhs, Rhs.M);
	return MakeLWCMatrix( ((Rhs).Tile) , ResultMatrix);
}

FLWCInverseMatrix LWCMultiply(FLWCInverseMatrix Lhs, float4x4 Rhs)
{
	float4x4 ResultMatrix = mul(Lhs.M, Rhs);
	return MakeLWCInverseMatrix(- ((Lhs).Tile) , ResultMatrix);
}

float4x4 LWCMultiply(FLWCMatrix Lhs, FLWCInverseMatrix Rhs)
{

	float4x4 Result = Lhs.M;
	Result = mul(Result, MakeTranslationMatrix(( ((Lhs).Tile)  +  ((Rhs).Tile) ) *  2097152.00f ));
	Result = mul(Result, Rhs.M);
	return Result;
}

float4x4 LWCMultiplyTranslation(FLWCMatrix Lhs, FLWCVector3 Rhs)
{
	float4x4 Result = Lhs.M;
	Result[3].xyz += ( ((Lhs).Tile)  +  ((Rhs).Tile) ) *  2097152.00f ;
	Result[3].xyz += Rhs.Offset;
	return Result;
}

FLWCMatrix LWCMultiplyTranslation(float4x4 Lhs, FLWCVector3 Rhs)
{
	FLWCMatrix Result = MakeLWCMatrix( ((Rhs).Tile) , Lhs);
	Result.M[3].xyz += Rhs.Offset;
	return Result;
}

float4x4 LWCMultiplyTranslation(FLWCVector3 Lhs, FLWCInverseMatrix Rhs)
{
	float3 Offset = ( ((Lhs).Tile)  +  ((Rhs).Tile) ) *  2097152.00f  + Lhs.Offset;
	return mul(MakeTranslationMatrix(Offset), Rhs.M);
}

FLWCInverseMatrix LWCMultiplyTranslation(FLWCVector3 Lhs, float4x4 Rhs)
{
	FLWCInverseMatrix Result = MakeLWCInverseMatrix(- ((Lhs).Tile) , Rhs);
	Result.M = mul(MakeTranslationMatrix(Lhs.Offset), Result.M);
	return Result;
}

FLWCVector3 LWCGetOrigin(FLWCMatrix InMatrix)
{
	return MakeLWCVector3( ((InMatrix).Tile) , InMatrix.M[3].xyz);
}

void LWCSetOrigin(inout FLWCMatrix InOutMatrix, FLWCVector3 Origin)
{
	(InOutMatrix).Tile = ( ((Origin).Tile) ) ;
	InOutMatrix.M[3].xyz = Origin.Offset;
}
#line 335 "/Engine/Private/LargeWorldCoordinates.ush"
#line 1 "LWCOperations.ush"




FLWCScalar  LWCNormalizeTile( FLWCScalar  V)
{
	float  IntTile = floor(V.Tile + (V.Offset *  4.76837158e-07f  + 0.5f));
	return  MakeLWCScalar (IntTile, (V.Tile - IntTile) *  2097152.00f  + V.Offset);
}


FLWCScalar  LWCMakeIntTile( FLWCScalar  V)
{
	float  IntTile = floor(V.Tile);
	return  MakeLWCScalar (IntTile, (V.Tile - IntTile) *  2097152.00f  + V.Offset);
}

float  LWCSqrtUnscaled( FLWCScalar  V) { return sqrt(V.Offset *  4.76837158e-07f  +  ((V).Tile) ); }
float  LWCRsqrtUnscaled( FLWCScalar  V) { return rsqrt(V.Offset *  4.76837158e-07f  +  ((V).Tile) ); }
float  LWCRcpUnscaled( FLWCScalar  V) { return rcp(V.Offset *  4.76837158e-07f  +  ((V).Tile) ); }
float  LWCSqrtScaled( FLWCScalar  V, float Scale) { return LWCSqrtUnscaled(V) * Scale; }
float  LWCRsqrtScaled( FLWCScalar  V, float Scale) { return LWCRsqrtUnscaled(V) * Scale; }
float  LWCRcpScaled( FLWCScalar  V, float Scale) { return LWCRcpUnscaled(V) * Scale; }
float  LWCSqrt( FLWCScalar  V) { return LWCSqrtScaled(V,  1448.15466f ); }
float  LWCRsqrt( FLWCScalar  V) { return LWCRsqrtScaled(V,  0.000690533954f ); }
float  LWCRcp( FLWCScalar  V) { return LWCRcpScaled(V,  4.76837158e-07f ); }
#line 36 "/Engine/Private/LWCOperations.ush"
bool LWCGreater( FLWCScalar Lhs, FLWCScalar Rhs) { return ( ((Lhs).Tile) - ((Rhs).Tile) ) * 2097152.00f > Rhs.Offset - Lhs.Offset; } bool LWCGreater( float Lhs, FLWCScalar Rhs) { return - ((Rhs).Tile) * 2097152.00f > Rhs.Offset - Lhs; } bool LWCGreater( FLWCScalar Lhs, float Rhs) { return ((Lhs).Tile) * 2097152.00f > Rhs - Lhs.Offset; }
bool LWCGreaterEqual( FLWCScalar Lhs, FLWCScalar Rhs) { return ( ((Lhs).Tile) - ((Rhs).Tile) ) * 2097152.00f >= Rhs.Offset - Lhs.Offset; } bool LWCGreaterEqual( float Lhs, FLWCScalar Rhs) { return - ((Rhs).Tile) * 2097152.00f >= Rhs.Offset - Lhs; } bool LWCGreaterEqual( FLWCScalar Lhs, float Rhs) { return ((Lhs).Tile) * 2097152.00f >= Rhs - Lhs.Offset; }
bool LWCLess( FLWCScalar Lhs, FLWCScalar Rhs) { return ( ((Lhs).Tile) - ((Rhs).Tile) ) * 2097152.00f < Rhs.Offset - Lhs.Offset; } bool LWCLess( float Lhs, FLWCScalar Rhs) { return - ((Rhs).Tile) * 2097152.00f < Rhs.Offset - Lhs; } bool LWCLess( FLWCScalar Lhs, float Rhs) { return ((Lhs).Tile) * 2097152.00f < Rhs - Lhs.Offset; }
bool LWCLessEqual( FLWCScalar Lhs, FLWCScalar Rhs) { return ( ((Lhs).Tile) - ((Rhs).Tile) ) * 2097152.00f <= Rhs.Offset - Lhs.Offset; } bool LWCLessEqual( float Lhs, FLWCScalar Rhs) { return - ((Rhs).Tile) * 2097152.00f <= Rhs.Offset - Lhs; } bool LWCLessEqual( FLWCScalar Lhs, float Rhs) { return ((Lhs).Tile) * 2097152.00f <= Rhs - Lhs.Offset; }





float  LWCDdx( FLWCScalar  V) { return ( float )0; }
float  LWCDdy( FLWCScalar  V) { return ( float )0; }
#line 52 "/Engine/Private/LWCOperations.ush"
FLWCScalar  LWCAdd( FLWCScalar  Lhs,  FLWCScalar  Rhs) { return  MakeLWCScalar ( ((Lhs).Tile)  +  ((Rhs).Tile) , Lhs.Offset + Rhs.Offset); }
FLWCScalar  LWCAdd( float  Lhs,  FLWCScalar  Rhs) { return  MakeLWCScalar ( ((Rhs).Tile) , Lhs + Rhs.Offset); }
FLWCScalar  LWCAdd( FLWCScalar  Lhs,  float  Rhs) { return  MakeLWCScalar ( ((Lhs).Tile) , Lhs.Offset + Rhs); }

FLWCScalar  LWCSubtract( FLWCScalar  Lhs,  FLWCScalar  Rhs) { return  MakeLWCScalar ( ((Lhs).Tile)  -  ((Rhs).Tile) , Lhs.Offset - Rhs.Offset); }
FLWCScalar  LWCSubtract( float  Lhs,  FLWCScalar  Rhs) { return  MakeLWCScalar (- ((Rhs).Tile) , Lhs - Rhs.Offset); }
FLWCScalar  LWCSubtract( FLWCScalar  Lhs,  float  Rhs) { return  MakeLWCScalar ( ((Lhs).Tile) , Lhs.Offset - Rhs); }

bool  LWCEquals( FLWCScalar  Lhs,  FLWCScalar  Rhs)
{
	return ( ((Lhs).Tile)  -  ((Rhs).Tile) ) *  2097152.00f  == Rhs.Offset - Lhs.Offset;
}
bool  LWCEquals( float  Lhs,  FLWCScalar  Rhs)
{
	return - ((Rhs).Tile)  *  2097152.00f  == Rhs.Offset - Lhs;
}
bool  LWCEquals( FLWCScalar  Lhs,  float  Rhs)
{
	return  ((Lhs).Tile)  *  2097152.00f  == Rhs - Lhs.Offset;
}
bool  LWCEqualsApprox( FLWCScalar  Lhs,  FLWCScalar  Rhs, float Threshold)
{
	return abs(( ((Lhs).Tile)  -  ((Rhs).Tile) ) *  2097152.00f  + (Lhs.Offset - Rhs.Offset)) < ( float )Threshold;
}
bool  LWCEqualsApprox( float  Lhs,  FLWCScalar  Rhs, float Threshold)
{
	return abs(- ((Rhs).Tile)  *  2097152.00f  + (Lhs - Rhs.Offset)) < ( float )Threshold;
}
bool  LWCEqualsApprox( FLWCScalar  Lhs,  float  Rhs, float Threshold)
{
	return abs( ((Lhs).Tile)  *  2097152.00f  + (Lhs.Offset - Rhs)) < ( float )Threshold;
}

FLWCScalar  LWCSelect( bool  S,  FLWCScalar  Lhs,  FLWCScalar  Rhs) { return  MakeLWCScalar (S ?  ((Lhs).Tile)  :  ((Rhs).Tile) , S ? Lhs.Offset : Rhs.Offset); }
FLWCScalar  LWCSelect( bool  S,  float  Lhs,  FLWCScalar  Rhs) { return  MakeLWCScalar (S ? ( float )0 :  ((Rhs).Tile) , S ? Lhs : Rhs.Offset); }
FLWCScalar  LWCSelect( bool  S,  FLWCScalar  Lhs,  float  Rhs) { return  MakeLWCScalar (S ?  ((Lhs).Tile)  : ( float )0, S ? Lhs.Offset : Rhs); }

FLWCScalar  LWCNegate( FLWCScalar  V) { return  MakeLWCScalar (- ((V).Tile) , -V.Offset); }

float  LWCFrac( FLWCScalar  V)
{
	float  FracTile = frac( ((V).Tile)  *  2097152.00f );
	return frac(FracTile + V.Offset);
}

FLWCScalar  LWCFloor( FLWCScalar  V) {  FLWCScalar  VN = LWCMakeIntTile(V); return  MakeLWCScalar ( ((VN).Tile) , floor(VN.Offset)); }
FLWCScalar  LWCCeil( FLWCScalar  V) {  FLWCScalar  VN = LWCMakeIntTile(V); return  MakeLWCScalar ( ((VN).Tile) , ceil(VN.Offset)); }
FLWCScalar  LWCRound( FLWCScalar  V) {  FLWCScalar  VN = LWCMakeIntTile(V); return  MakeLWCScalar ( ((VN).Tile) , round(VN.Offset)); }
FLWCScalar  LWCTrunc( FLWCScalar  V) {  FLWCScalar  VN = LWCMakeIntTile(V); return  MakeLWCScalar ( ((VN).Tile) , trunc(VN.Offset)); }


float  LWCSign( FLWCScalar  V) { return  float (sign(LWCToFloat(V))); }
float  LWCSaturate( FLWCScalar  V) { return saturate(LWCToFloat(V)); }
float  LWCClampScalar( FLWCScalar  V, float Low, float High) { return clamp(LWCToFloat(V), Low, High); }

FLWCScalar  LWCMultiply( FLWCScalar  Lhs,  FLWCScalar  Rhs)
{
	return  MakeLWCScalar ( ((Lhs).Tile)  * ( ((Rhs).Tile)  *  2097152.00f  + Rhs.Offset) +  ((Rhs).Tile)  * Lhs.Offset, Lhs.Offset * Rhs.Offset);
}
FLWCScalar  LWCMultiply( float  Lhs,  FLWCScalar  Rhs) { return  MakeLWCScalar ( ((Rhs).Tile)  * Lhs, Lhs * Rhs.Offset); }
FLWCScalar  LWCMultiply( FLWCScalar  Lhs,  float  Rhs) { return  MakeLWCScalar ( ((Lhs).Tile)  * Rhs, Lhs.Offset * Rhs); }

FLWCScalar  LWCDivide( FLWCScalar  Lhs,  FLWCScalar  Rhs) { return LWCMultiply(Lhs, LWCRcp(Rhs)); }
FLWCScalar  LWCDivide( FLWCScalar  Lhs,  float  Rhs) { return LWCMultiply(Lhs, rcp(Rhs)); }
FLWCScalar  LWCDivide( float  Lhs,  FLWCScalar  Rhs) { return  MakeLWCScalar (( float )0, Lhs * LWCRcp(Rhs)); }


FLWCScalar  LWCLerp( FLWCScalar  Lhs,  FLWCScalar  Rhs,  float  S)
{
	return  MakeLWCScalar (lerp( ((Lhs).Tile) ,  ((Rhs).Tile) , S), lerp(Lhs.Offset, Rhs.Offset, S));
}

float  LWCFmod( FLWCScalar  Lhs,  float  Rhs)
{
	return LWCToFloat(LWCSubtract(Lhs, LWCMultiply(LWCTrunc(LWCDivide(Lhs, Rhs)), Rhs)));


}
float  LWCFmodFloor( FLWCScalar  Lhs,  float  Rhs)
{
	return LWCToFloat(LWCSubtract(Lhs, LWCMultiply(LWCFloor(LWCDivide(Lhs, Rhs)), Rhs)));


}
float  LWCFmodFloorPI( FLWCScalar  V)
{
	return LWCFmodFloor(V, PI);

}
float  LWCFmodFloor2PI( FLWCScalar  V)
{
	return LWCFmodFloor(V, 2.0f * PI);

}

float  LWCSin( FLWCScalar  V) { return sin(LWCFmodFloor2PI(V)); }
float  LWCCos( FLWCScalar  V) { return cos(LWCFmodFloor2PI(V)); }
float  LWCTan( FLWCScalar  V) { return tan(LWCFmodFloorPI(V)); }
float  LWCASin( FLWCScalar  V) { return asin(LWCClampScalar(V, -1.0f, 1.0f)); }
float  LWCACos( FLWCScalar  V) { return acos(LWCClampScalar(V, -1.0f, 1.0f)); }
float  LWCATan( FLWCScalar  V) { return atan(LWCClampScalar(V, -0.5f*PI, 0.5f*PI)); }

float  LWCSmoothStep( FLWCScalar  Lhs,  FLWCScalar  Rhs,  FLWCScalar  S)
{
	float  t = LWCSaturate(LWCDivide(LWCSubtract(S, Lhs), LWCSubtract(Rhs, Lhs)));
	return t*t*(3.0f - (2.0f*t));
}

FLWCScalar  LWCMin( FLWCScalar  Lhs,  FLWCScalar  Rhs) { return LWCSelect(LWCLess(Lhs, Rhs), Lhs, Rhs); }
FLWCScalar  LWCMin( float  Lhs,  FLWCScalar  Rhs) { return LWCSelect(LWCLess(Lhs, Rhs), Lhs, Rhs); }
FLWCScalar  LWCMin( FLWCScalar  Lhs,  float  Rhs) { return LWCSelect(LWCLess(Lhs, Rhs), Lhs, Rhs); }
FLWCScalar  LWCMax( FLWCScalar  Lhs,  FLWCScalar  Rhs) { return LWCSelect(LWCGreater(Lhs, Rhs), Lhs, Rhs); }
FLWCScalar  LWCMax( float  Lhs,  FLWCScalar  Rhs) { return LWCSelect(LWCGreater(Lhs, Rhs), Lhs, Rhs); }
FLWCScalar  LWCMax( FLWCScalar  Lhs,  float  Rhs) { return LWCSelect(LWCGreater(Lhs, Rhs), Lhs, Rhs); }

FLWCScalar  LWCAbs( FLWCScalar  V) { return LWCSelect(LWCLess(V, ( float )0), LWCNegate(V), V); }

float  LWCStep( FLWCScalar  Lhs,  FLWCScalar  Rhs) { return LWCGreaterEqual(Rhs, Lhs) ? ( float )1.0f : ( float )0.0f; }
float  LWCStep( FLWCScalar  Lhs,  float  Rhs) { return LWCGreaterEqual(Rhs, Lhs) ? ( float )1.0f : ( float )0.0f; }
float  LWCStep( float  Lhs,  FLWCScalar  Rhs) { return LWCGreaterEqual(Rhs, Lhs) ? ( float )1.0f : ( float )0.0f; }


FLWCScalar  LWCSquareScaled( FLWCScalar  V)
{
	float  OffsetScaled = V.Offset *  4.76837158e-07f ;
	return  MakeLWCScalar ( ((V).Tile)  * ( ((V).Tile)  + OffsetScaled * 2.0f), V.Offset * OffsetScaled);
}
#line 336 "/Engine/Private/LargeWorldCoordinates.ush"
#line 345 "/Engine/Private/LargeWorldCoordinates.ush"
#line 1 "LWCOperations.ush"




FLWCVector2  LWCNormalizeTile( FLWCVector2  V)
{
	float2  IntTile = floor(V.Tile + (V.Offset *  4.76837158e-07f  + 0.5f));
	return  MakeLWCVector2 (IntTile, (V.Tile - IntTile) *  2097152.00f  + V.Offset);
}


FLWCVector2  LWCMakeIntTile( FLWCVector2  V)
{
	float2  IntTile = floor(V.Tile);
	return  MakeLWCVector2 (IntTile, (V.Tile - IntTile) *  2097152.00f  + V.Offset);
}

float2  LWCSqrtUnscaled( FLWCVector2  V) { return sqrt(V.Offset *  4.76837158e-07f  +  ((V).Tile) ); }
float2  LWCRsqrtUnscaled( FLWCVector2  V) { return rsqrt(V.Offset *  4.76837158e-07f  +  ((V).Tile) ); }
float2  LWCRcpUnscaled( FLWCVector2  V) { return rcp(V.Offset *  4.76837158e-07f  +  ((V).Tile) ); }
float2  LWCSqrtScaled( FLWCVector2  V, float Scale) { return LWCSqrtUnscaled(V) * Scale; }
float2  LWCRsqrtScaled( FLWCVector2  V, float Scale) { return LWCRsqrtUnscaled(V) * Scale; }
float2  LWCRcpScaled( FLWCVector2  V, float Scale) { return LWCRcpUnscaled(V) * Scale; }
float2  LWCSqrt( FLWCVector2  V) { return LWCSqrtScaled(V,  1448.15466f ); }
float2  LWCRsqrt( FLWCVector2  V) { return LWCRsqrtScaled(V,  0.000690533954f ); }
float2  LWCRcp( FLWCVector2  V) { return LWCRcpScaled(V,  4.76837158e-07f ); }
#line 36 "/Engine/Private/LWCOperations.ush"
bool2 LWCGreater( FLWCVector2 Lhs, FLWCVector2 Rhs) { return ( ((Lhs).Tile) - ((Rhs).Tile) ) * 2097152.00f > Rhs.Offset - Lhs.Offset; } bool2 LWCGreater( float2 Lhs, FLWCVector2 Rhs) { return - ((Rhs).Tile) * 2097152.00f > Rhs.Offset - Lhs; } bool2 LWCGreater( FLWCVector2 Lhs, float2 Rhs) { return ((Lhs).Tile) * 2097152.00f > Rhs - Lhs.Offset; }
bool2 LWCGreaterEqual( FLWCVector2 Lhs, FLWCVector2 Rhs) { return ( ((Lhs).Tile) - ((Rhs).Tile) ) * 2097152.00f >= Rhs.Offset - Lhs.Offset; } bool2 LWCGreaterEqual( float2 Lhs, FLWCVector2 Rhs) { return - ((Rhs).Tile) * 2097152.00f >= Rhs.Offset - Lhs; } bool2 LWCGreaterEqual( FLWCVector2 Lhs, float2 Rhs) { return ((Lhs).Tile) * 2097152.00f >= Rhs - Lhs.Offset; }
bool2 LWCLess( FLWCVector2 Lhs, FLWCVector2 Rhs) { return ( ((Lhs).Tile) - ((Rhs).Tile) ) * 2097152.00f < Rhs.Offset - Lhs.Offset; } bool2 LWCLess( float2 Lhs, FLWCVector2 Rhs) { return - ((Rhs).Tile) * 2097152.00f < Rhs.Offset - Lhs; } bool2 LWCLess( FLWCVector2 Lhs, float2 Rhs) { return ((Lhs).Tile) * 2097152.00f < Rhs - Lhs.Offset; }
bool2 LWCLessEqual( FLWCVector2 Lhs, FLWCVector2 Rhs) { return ( ((Lhs).Tile) - ((Rhs).Tile) ) * 2097152.00f <= Rhs.Offset - Lhs.Offset; } bool2 LWCLessEqual( float2 Lhs, FLWCVector2 Rhs) { return - ((Rhs).Tile) * 2097152.00f <= Rhs.Offset - Lhs; } bool2 LWCLessEqual( FLWCVector2 Lhs, float2 Rhs) { return ((Lhs).Tile) * 2097152.00f <= Rhs - Lhs.Offset; }





float2  LWCDdx( FLWCVector2  V) { return ( float2 )0; }
float2  LWCDdy( FLWCVector2  V) { return ( float2 )0; }
#line 52 "/Engine/Private/LWCOperations.ush"
FLWCVector2  LWCAdd( FLWCVector2  Lhs,  FLWCVector2  Rhs) { return  MakeLWCVector2 ( ((Lhs).Tile)  +  ((Rhs).Tile) , Lhs.Offset + Rhs.Offset); }
FLWCVector2  LWCAdd( float2  Lhs,  FLWCVector2  Rhs) { return  MakeLWCVector2 ( ((Rhs).Tile) , Lhs + Rhs.Offset); }
FLWCVector2  LWCAdd( FLWCVector2  Lhs,  float2  Rhs) { return  MakeLWCVector2 ( ((Lhs).Tile) , Lhs.Offset + Rhs); }

FLWCVector2  LWCSubtract( FLWCVector2  Lhs,  FLWCVector2  Rhs) { return  MakeLWCVector2 ( ((Lhs).Tile)  -  ((Rhs).Tile) , Lhs.Offset - Rhs.Offset); }
FLWCVector2  LWCSubtract( float2  Lhs,  FLWCVector2  Rhs) { return  MakeLWCVector2 (- ((Rhs).Tile) , Lhs - Rhs.Offset); }
FLWCVector2  LWCSubtract( FLWCVector2  Lhs,  float2  Rhs) { return  MakeLWCVector2 ( ((Lhs).Tile) , Lhs.Offset - Rhs); }

bool2  LWCEquals( FLWCVector2  Lhs,  FLWCVector2  Rhs)
{
	return ( ((Lhs).Tile)  -  ((Rhs).Tile) ) *  2097152.00f  == Rhs.Offset - Lhs.Offset;
}
bool2  LWCEquals( float2  Lhs,  FLWCVector2  Rhs)
{
	return - ((Rhs).Tile)  *  2097152.00f  == Rhs.Offset - Lhs;
}
bool2  LWCEquals( FLWCVector2  Lhs,  float2  Rhs)
{
	return  ((Lhs).Tile)  *  2097152.00f  == Rhs - Lhs.Offset;
}
bool2  LWCEqualsApprox( FLWCVector2  Lhs,  FLWCVector2  Rhs, float Threshold)
{
	return abs(( ((Lhs).Tile)  -  ((Rhs).Tile) ) *  2097152.00f  + (Lhs.Offset - Rhs.Offset)) < ( float2 )Threshold;
}
bool2  LWCEqualsApprox( float2  Lhs,  FLWCVector2  Rhs, float Threshold)
{
	return abs(- ((Rhs).Tile)  *  2097152.00f  + (Lhs - Rhs.Offset)) < ( float2 )Threshold;
}
bool2  LWCEqualsApprox( FLWCVector2  Lhs,  float2  Rhs, float Threshold)
{
	return abs( ((Lhs).Tile)  *  2097152.00f  + (Lhs.Offset - Rhs)) < ( float2 )Threshold;
}

FLWCVector2  LWCSelect( bool2  S,  FLWCVector2  Lhs,  FLWCVector2  Rhs) { return  MakeLWCVector2 (S ?  ((Lhs).Tile)  :  ((Rhs).Tile) , S ? Lhs.Offset : Rhs.Offset); }
FLWCVector2  LWCSelect( bool2  S,  float2  Lhs,  FLWCVector2  Rhs) { return  MakeLWCVector2 (S ? ( float2 )0 :  ((Rhs).Tile) , S ? Lhs : Rhs.Offset); }
FLWCVector2  LWCSelect( bool2  S,  FLWCVector2  Lhs,  float2  Rhs) { return  MakeLWCVector2 (S ?  ((Lhs).Tile)  : ( float2 )0, S ? Lhs.Offset : Rhs); }

FLWCVector2  LWCNegate( FLWCVector2  V) { return  MakeLWCVector2 (- ((V).Tile) , -V.Offset); }

float2  LWCFrac( FLWCVector2  V)
{
	float2  FracTile = frac( ((V).Tile)  *  2097152.00f );
	return frac(FracTile + V.Offset);
}

FLWCVector2  LWCFloor( FLWCVector2  V) {  FLWCVector2  VN = LWCMakeIntTile(V); return  MakeLWCVector2 ( ((VN).Tile) , floor(VN.Offset)); }
FLWCVector2  LWCCeil( FLWCVector2  V) {  FLWCVector2  VN = LWCMakeIntTile(V); return  MakeLWCVector2 ( ((VN).Tile) , ceil(VN.Offset)); }
FLWCVector2  LWCRound( FLWCVector2  V) {  FLWCVector2  VN = LWCMakeIntTile(V); return  MakeLWCVector2 ( ((VN).Tile) , round(VN.Offset)); }
FLWCVector2  LWCTrunc( FLWCVector2  V) {  FLWCVector2  VN = LWCMakeIntTile(V); return  MakeLWCVector2 ( ((VN).Tile) , trunc(VN.Offset)); }


float2  LWCSign( FLWCVector2  V) { return  float2 (sign(LWCToFloat(V))); }
float2  LWCSaturate( FLWCVector2  V) { return saturate(LWCToFloat(V)); }
float2  LWCClampScalar( FLWCVector2  V, float Low, float High) { return clamp(LWCToFloat(V), Low, High); }

FLWCVector2  LWCMultiply( FLWCVector2  Lhs,  FLWCVector2  Rhs)
{
	return  MakeLWCVector2 ( ((Lhs).Tile)  * ( ((Rhs).Tile)  *  2097152.00f  + Rhs.Offset) +  ((Rhs).Tile)  * Lhs.Offset, Lhs.Offset * Rhs.Offset);
}
FLWCVector2  LWCMultiply( float2  Lhs,  FLWCVector2  Rhs) { return  MakeLWCVector2 ( ((Rhs).Tile)  * Lhs, Lhs * Rhs.Offset); }
FLWCVector2  LWCMultiply( FLWCVector2  Lhs,  float2  Rhs) { return  MakeLWCVector2 ( ((Lhs).Tile)  * Rhs, Lhs.Offset * Rhs); }

FLWCVector2  LWCDivide( FLWCVector2  Lhs,  FLWCVector2  Rhs) { return LWCMultiply(Lhs, LWCRcp(Rhs)); }
FLWCVector2  LWCDivide( FLWCVector2  Lhs,  float2  Rhs) { return LWCMultiply(Lhs, rcp(Rhs)); }
FLWCVector2  LWCDivide( float2  Lhs,  FLWCVector2  Rhs) { return  MakeLWCVector2 (( float2 )0, Lhs * LWCRcp(Rhs)); }


FLWCVector2  LWCLerp( FLWCVector2  Lhs,  FLWCVector2  Rhs,  float2  S)
{
	return  MakeLWCVector2 (lerp( ((Lhs).Tile) ,  ((Rhs).Tile) , S), lerp(Lhs.Offset, Rhs.Offset, S));
}

float2  LWCFmod( FLWCVector2  Lhs,  float2  Rhs)
{
	return LWCToFloat(LWCSubtract(Lhs, LWCMultiply(LWCTrunc(LWCDivide(Lhs, Rhs)), Rhs)));


}
float2  LWCFmodFloor( FLWCVector2  Lhs,  float2  Rhs)
{
	return LWCToFloat(LWCSubtract(Lhs, LWCMultiply(LWCFloor(LWCDivide(Lhs, Rhs)), Rhs)));


}
float2  LWCFmodFloorPI( FLWCVector2  V)
{
	return LWCFmodFloor(V, PI);

}
float2  LWCFmodFloor2PI( FLWCVector2  V)
{
	return LWCFmodFloor(V, 2.0f * PI);

}

float2  LWCSin( FLWCVector2  V) { return sin(LWCFmodFloor2PI(V)); }
float2  LWCCos( FLWCVector2  V) { return cos(LWCFmodFloor2PI(V)); }
float2  LWCTan( FLWCVector2  V) { return tan(LWCFmodFloorPI(V)); }
float2  LWCASin( FLWCVector2  V) { return asin(LWCClampScalar(V, -1.0f, 1.0f)); }
float2  LWCACos( FLWCVector2  V) { return acos(LWCClampScalar(V, -1.0f, 1.0f)); }
float2  LWCATan( FLWCVector2  V) { return atan(LWCClampScalar(V, -0.5f*PI, 0.5f*PI)); }

float2  LWCSmoothStep( FLWCVector2  Lhs,  FLWCVector2  Rhs,  FLWCVector2  S)
{
	float2  t = LWCSaturate(LWCDivide(LWCSubtract(S, Lhs), LWCSubtract(Rhs, Lhs)));
	return t*t*(3.0f - (2.0f*t));
}

FLWCVector2  LWCMin( FLWCVector2  Lhs,  FLWCVector2  Rhs) { return LWCSelect(LWCLess(Lhs, Rhs), Lhs, Rhs); }
FLWCVector2  LWCMin( float2  Lhs,  FLWCVector2  Rhs) { return LWCSelect(LWCLess(Lhs, Rhs), Lhs, Rhs); }
FLWCVector2  LWCMin( FLWCVector2  Lhs,  float2  Rhs) { return LWCSelect(LWCLess(Lhs, Rhs), Lhs, Rhs); }
FLWCVector2  LWCMax( FLWCVector2  Lhs,  FLWCVector2  Rhs) { return LWCSelect(LWCGreater(Lhs, Rhs), Lhs, Rhs); }
FLWCVector2  LWCMax( float2  Lhs,  FLWCVector2  Rhs) { return LWCSelect(LWCGreater(Lhs, Rhs), Lhs, Rhs); }
FLWCVector2  LWCMax( FLWCVector2  Lhs,  float2  Rhs) { return LWCSelect(LWCGreater(Lhs, Rhs), Lhs, Rhs); }

FLWCVector2  LWCAbs( FLWCVector2  V) { return LWCSelect(LWCLess(V, ( float2 )0), LWCNegate(V), V); }

float2  LWCStep( FLWCVector2  Lhs,  FLWCVector2  Rhs) { return LWCGreaterEqual(Rhs, Lhs) ? ( float2 )1.0f : ( float2 )0.0f; }
float2  LWCStep( FLWCVector2  Lhs,  float2  Rhs) { return LWCGreaterEqual(Rhs, Lhs) ? ( float2 )1.0f : ( float2 )0.0f; }
float2  LWCStep( float2  Lhs,  FLWCVector2  Rhs) { return LWCGreaterEqual(Rhs, Lhs) ? ( float2 )1.0f : ( float2 )0.0f; }


FLWCVector2  LWCSquareScaled( FLWCVector2  V)
{
	float2  OffsetScaled = V.Offset *  4.76837158e-07f ;
	return  MakeLWCVector2 ( ((V).Tile)  * ( ((V).Tile)  + OffsetScaled * 2.0f), V.Offset * OffsetScaled);
}
#line 346 "/Engine/Private/LargeWorldCoordinates.ush"
#line 355 "/Engine/Private/LargeWorldCoordinates.ush"
#line 1 "LWCOperations.ush"




FLWCVector3  LWCNormalizeTile( FLWCVector3  V)
{
	float3  IntTile = floor(V.Tile + (V.Offset *  4.76837158e-07f  + 0.5f));
	return  MakeLWCVector3 (IntTile, (V.Tile - IntTile) *  2097152.00f  + V.Offset);
}


FLWCVector3  LWCMakeIntTile( FLWCVector3  V)
{
	float3  IntTile = floor(V.Tile);
	return  MakeLWCVector3 (IntTile, (V.Tile - IntTile) *  2097152.00f  + V.Offset);
}

float3  LWCSqrtUnscaled( FLWCVector3  V) { return sqrt(V.Offset *  4.76837158e-07f  +  ((V).Tile) ); }
float3  LWCRsqrtUnscaled( FLWCVector3  V) { return rsqrt(V.Offset *  4.76837158e-07f  +  ((V).Tile) ); }
float3  LWCRcpUnscaled( FLWCVector3  V) { return rcp(V.Offset *  4.76837158e-07f  +  ((V).Tile) ); }
float3  LWCSqrtScaled( FLWCVector3  V, float Scale) { return LWCSqrtUnscaled(V) * Scale; }
float3  LWCRsqrtScaled( FLWCVector3  V, float Scale) { return LWCRsqrtUnscaled(V) * Scale; }
float3  LWCRcpScaled( FLWCVector3  V, float Scale) { return LWCRcpUnscaled(V) * Scale; }
float3  LWCSqrt( FLWCVector3  V) { return LWCSqrtScaled(V,  1448.15466f ); }
float3  LWCRsqrt( FLWCVector3  V) { return LWCRsqrtScaled(V,  0.000690533954f ); }
float3  LWCRcp( FLWCVector3  V) { return LWCRcpScaled(V,  4.76837158e-07f ); }
#line 36 "/Engine/Private/LWCOperations.ush"
bool3 LWCGreater( FLWCVector3 Lhs, FLWCVector3 Rhs) { return ( ((Lhs).Tile) - ((Rhs).Tile) ) * 2097152.00f > Rhs.Offset - Lhs.Offset; } bool3 LWCGreater( float3 Lhs, FLWCVector3 Rhs) { return - ((Rhs).Tile) * 2097152.00f > Rhs.Offset - Lhs; } bool3 LWCGreater( FLWCVector3 Lhs, float3 Rhs) { return ((Lhs).Tile) * 2097152.00f > Rhs - Lhs.Offset; }
bool3 LWCGreaterEqual( FLWCVector3 Lhs, FLWCVector3 Rhs) { return ( ((Lhs).Tile) - ((Rhs).Tile) ) * 2097152.00f >= Rhs.Offset - Lhs.Offset; } bool3 LWCGreaterEqual( float3 Lhs, FLWCVector3 Rhs) { return - ((Rhs).Tile) * 2097152.00f >= Rhs.Offset - Lhs; } bool3 LWCGreaterEqual( FLWCVector3 Lhs, float3 Rhs) { return ((Lhs).Tile) * 2097152.00f >= Rhs - Lhs.Offset; }
bool3 LWCLess( FLWCVector3 Lhs, FLWCVector3 Rhs) { return ( ((Lhs).Tile) - ((Rhs).Tile) ) * 2097152.00f < Rhs.Offset - Lhs.Offset; } bool3 LWCLess( float3 Lhs, FLWCVector3 Rhs) { return - ((Rhs).Tile) * 2097152.00f < Rhs.Offset - Lhs; } bool3 LWCLess( FLWCVector3 Lhs, float3 Rhs) { return ((Lhs).Tile) * 2097152.00f < Rhs - Lhs.Offset; }
bool3 LWCLessEqual( FLWCVector3 Lhs, FLWCVector3 Rhs) { return ( ((Lhs).Tile) - ((Rhs).Tile) ) * 2097152.00f <= Rhs.Offset - Lhs.Offset; } bool3 LWCLessEqual( float3 Lhs, FLWCVector3 Rhs) { return - ((Rhs).Tile) * 2097152.00f <= Rhs.Offset - Lhs; } bool3 LWCLessEqual( FLWCVector3 Lhs, float3 Rhs) { return ((Lhs).Tile) * 2097152.00f <= Rhs - Lhs.Offset; }





float3  LWCDdx( FLWCVector3  V) { return ( float3 )0; }
float3  LWCDdy( FLWCVector3  V) { return ( float3 )0; }
#line 52 "/Engine/Private/LWCOperations.ush"
FLWCVector3  LWCAdd( FLWCVector3  Lhs,  FLWCVector3  Rhs) { return  MakeLWCVector3 ( ((Lhs).Tile)  +  ((Rhs).Tile) , Lhs.Offset + Rhs.Offset); }
FLWCVector3  LWCAdd( float3  Lhs,  FLWCVector3  Rhs) { return  MakeLWCVector3 ( ((Rhs).Tile) , Lhs + Rhs.Offset); }
FLWCVector3  LWCAdd( FLWCVector3  Lhs,  float3  Rhs) { return  MakeLWCVector3 ( ((Lhs).Tile) , Lhs.Offset + Rhs); }

FLWCVector3  LWCSubtract( FLWCVector3  Lhs,  FLWCVector3  Rhs) { return  MakeLWCVector3 ( ((Lhs).Tile)  -  ((Rhs).Tile) , Lhs.Offset - Rhs.Offset); }
FLWCVector3  LWCSubtract( float3  Lhs,  FLWCVector3  Rhs) { return  MakeLWCVector3 (- ((Rhs).Tile) , Lhs - Rhs.Offset); }
FLWCVector3  LWCSubtract( FLWCVector3  Lhs,  float3  Rhs) { return  MakeLWCVector3 ( ((Lhs).Tile) , Lhs.Offset - Rhs); }

bool3  LWCEquals( FLWCVector3  Lhs,  FLWCVector3  Rhs)
{
	return ( ((Lhs).Tile)  -  ((Rhs).Tile) ) *  2097152.00f  == Rhs.Offset - Lhs.Offset;
}
bool3  LWCEquals( float3  Lhs,  FLWCVector3  Rhs)
{
	return - ((Rhs).Tile)  *  2097152.00f  == Rhs.Offset - Lhs;
}
bool3  LWCEquals( FLWCVector3  Lhs,  float3  Rhs)
{
	return  ((Lhs).Tile)  *  2097152.00f  == Rhs - Lhs.Offset;
}
bool3  LWCEqualsApprox( FLWCVector3  Lhs,  FLWCVector3  Rhs, float Threshold)
{
	return abs(( ((Lhs).Tile)  -  ((Rhs).Tile) ) *  2097152.00f  + (Lhs.Offset - Rhs.Offset)) < ( float3 )Threshold;
}
bool3  LWCEqualsApprox( float3  Lhs,  FLWCVector3  Rhs, float Threshold)
{
	return abs(- ((Rhs).Tile)  *  2097152.00f  + (Lhs - Rhs.Offset)) < ( float3 )Threshold;
}
bool3  LWCEqualsApprox( FLWCVector3  Lhs,  float3  Rhs, float Threshold)
{
	return abs( ((Lhs).Tile)  *  2097152.00f  + (Lhs.Offset - Rhs)) < ( float3 )Threshold;
}

FLWCVector3  LWCSelect( bool3  S,  FLWCVector3  Lhs,  FLWCVector3  Rhs) { return  MakeLWCVector3 (S ?  ((Lhs).Tile)  :  ((Rhs).Tile) , S ? Lhs.Offset : Rhs.Offset); }
FLWCVector3  LWCSelect( bool3  S,  float3  Lhs,  FLWCVector3  Rhs) { return  MakeLWCVector3 (S ? ( float3 )0 :  ((Rhs).Tile) , S ? Lhs : Rhs.Offset); }
FLWCVector3  LWCSelect( bool3  S,  FLWCVector3  Lhs,  float3  Rhs) { return  MakeLWCVector3 (S ?  ((Lhs).Tile)  : ( float3 )0, S ? Lhs.Offset : Rhs); }

FLWCVector3  LWCNegate( FLWCVector3  V) { return  MakeLWCVector3 (- ((V).Tile) , -V.Offset); }

float3  LWCFrac( FLWCVector3  V)
{
	float3  FracTile = frac( ((V).Tile)  *  2097152.00f );
	return frac(FracTile + V.Offset);
}

FLWCVector3  LWCFloor( FLWCVector3  V) {  FLWCVector3  VN = LWCMakeIntTile(V); return  MakeLWCVector3 ( ((VN).Tile) , floor(VN.Offset)); }
FLWCVector3  LWCCeil( FLWCVector3  V) {  FLWCVector3  VN = LWCMakeIntTile(V); return  MakeLWCVector3 ( ((VN).Tile) , ceil(VN.Offset)); }
FLWCVector3  LWCRound( FLWCVector3  V) {  FLWCVector3  VN = LWCMakeIntTile(V); return  MakeLWCVector3 ( ((VN).Tile) , round(VN.Offset)); }
FLWCVector3  LWCTrunc( FLWCVector3  V) {  FLWCVector3  VN = LWCMakeIntTile(V); return  MakeLWCVector3 ( ((VN).Tile) , trunc(VN.Offset)); }


float3  LWCSign( FLWCVector3  V) { return  float3 (sign(LWCToFloat(V))); }
float3  LWCSaturate( FLWCVector3  V) { return saturate(LWCToFloat(V)); }
float3  LWCClampScalar( FLWCVector3  V, float Low, float High) { return clamp(LWCToFloat(V), Low, High); }

FLWCVector3  LWCMultiply( FLWCVector3  Lhs,  FLWCVector3  Rhs)
{
	return  MakeLWCVector3 ( ((Lhs).Tile)  * ( ((Rhs).Tile)  *  2097152.00f  + Rhs.Offset) +  ((Rhs).Tile)  * Lhs.Offset, Lhs.Offset * Rhs.Offset);
}
FLWCVector3  LWCMultiply( float3  Lhs,  FLWCVector3  Rhs) { return  MakeLWCVector3 ( ((Rhs).Tile)  * Lhs, Lhs * Rhs.Offset); }
FLWCVector3  LWCMultiply( FLWCVector3  Lhs,  float3  Rhs) { return  MakeLWCVector3 ( ((Lhs).Tile)  * Rhs, Lhs.Offset * Rhs); }

FLWCVector3  LWCDivide( FLWCVector3  Lhs,  FLWCVector3  Rhs) { return LWCMultiply(Lhs, LWCRcp(Rhs)); }
FLWCVector3  LWCDivide( FLWCVector3  Lhs,  float3  Rhs) { return LWCMultiply(Lhs, rcp(Rhs)); }
FLWCVector3  LWCDivide( float3  Lhs,  FLWCVector3  Rhs) { return  MakeLWCVector3 (( float3 )0, Lhs * LWCRcp(Rhs)); }


FLWCVector3  LWCLerp( FLWCVector3  Lhs,  FLWCVector3  Rhs,  float3  S)
{
	return  MakeLWCVector3 (lerp( ((Lhs).Tile) ,  ((Rhs).Tile) , S), lerp(Lhs.Offset, Rhs.Offset, S));
}

float3  LWCFmod( FLWCVector3  Lhs,  float3  Rhs)
{
	return LWCToFloat(LWCSubtract(Lhs, LWCMultiply(LWCTrunc(LWCDivide(Lhs, Rhs)), Rhs)));


}
float3  LWCFmodFloor( FLWCVector3  Lhs,  float3  Rhs)
{
	return LWCToFloat(LWCSubtract(Lhs, LWCMultiply(LWCFloor(LWCDivide(Lhs, Rhs)), Rhs)));


}
float3  LWCFmodFloorPI( FLWCVector3  V)
{
	return LWCFmodFloor(V, PI);

}
float3  LWCFmodFloor2PI( FLWCVector3  V)
{
	return LWCFmodFloor(V, 2.0f * PI);

}

float3  LWCSin( FLWCVector3  V) { return sin(LWCFmodFloor2PI(V)); }
float3  LWCCos( FLWCVector3  V) { return cos(LWCFmodFloor2PI(V)); }
float3  LWCTan( FLWCVector3  V) { return tan(LWCFmodFloorPI(V)); }
float3  LWCASin( FLWCVector3  V) { return asin(LWCClampScalar(V, -1.0f, 1.0f)); }
float3  LWCACos( FLWCVector3  V) { return acos(LWCClampScalar(V, -1.0f, 1.0f)); }
float3  LWCATan( FLWCVector3  V) { return atan(LWCClampScalar(V, -0.5f*PI, 0.5f*PI)); }

float3  LWCSmoothStep( FLWCVector3  Lhs,  FLWCVector3  Rhs,  FLWCVector3  S)
{
	float3  t = LWCSaturate(LWCDivide(LWCSubtract(S, Lhs), LWCSubtract(Rhs, Lhs)));
	return t*t*(3.0f - (2.0f*t));
}

FLWCVector3  LWCMin( FLWCVector3  Lhs,  FLWCVector3  Rhs) { return LWCSelect(LWCLess(Lhs, Rhs), Lhs, Rhs); }
FLWCVector3  LWCMin( float3  Lhs,  FLWCVector3  Rhs) { return LWCSelect(LWCLess(Lhs, Rhs), Lhs, Rhs); }
FLWCVector3  LWCMin( FLWCVector3  Lhs,  float3  Rhs) { return LWCSelect(LWCLess(Lhs, Rhs), Lhs, Rhs); }
FLWCVector3  LWCMax( FLWCVector3  Lhs,  FLWCVector3  Rhs) { return LWCSelect(LWCGreater(Lhs, Rhs), Lhs, Rhs); }
FLWCVector3  LWCMax( float3  Lhs,  FLWCVector3  Rhs) { return LWCSelect(LWCGreater(Lhs, Rhs), Lhs, Rhs); }
FLWCVector3  LWCMax( FLWCVector3  Lhs,  float3  Rhs) { return LWCSelect(LWCGreater(Lhs, Rhs), Lhs, Rhs); }

FLWCVector3  LWCAbs( FLWCVector3  V) { return LWCSelect(LWCLess(V, ( float3 )0), LWCNegate(V), V); }

float3  LWCStep( FLWCVector3  Lhs,  FLWCVector3  Rhs) { return LWCGreaterEqual(Rhs, Lhs) ? ( float3 )1.0f : ( float3 )0.0f; }
float3  LWCStep( FLWCVector3  Lhs,  float3  Rhs) { return LWCGreaterEqual(Rhs, Lhs) ? ( float3 )1.0f : ( float3 )0.0f; }
float3  LWCStep( float3  Lhs,  FLWCVector3  Rhs) { return LWCGreaterEqual(Rhs, Lhs) ? ( float3 )1.0f : ( float3 )0.0f; }


FLWCVector3  LWCSquareScaled( FLWCVector3  V)
{
	float3  OffsetScaled = V.Offset *  4.76837158e-07f ;
	return  MakeLWCVector3 ( ((V).Tile)  * ( ((V).Tile)  + OffsetScaled * 2.0f), V.Offset * OffsetScaled);
}
#line 356 "/Engine/Private/LargeWorldCoordinates.ush"
#line 365 "/Engine/Private/LargeWorldCoordinates.ush"
#line 1 "LWCOperations.ush"




FLWCVector4  LWCNormalizeTile( FLWCVector4  V)
{
	float4  IntTile = floor(V.Tile + (V.Offset *  4.76837158e-07f  + 0.5f));
	return  MakeLWCVector4 (IntTile, (V.Tile - IntTile) *  2097152.00f  + V.Offset);
}


FLWCVector4  LWCMakeIntTile( FLWCVector4  V)
{
	float4  IntTile = floor(V.Tile);
	return  MakeLWCVector4 (IntTile, (V.Tile - IntTile) *  2097152.00f  + V.Offset);
}

float4  LWCSqrtUnscaled( FLWCVector4  V) { return sqrt(V.Offset *  4.76837158e-07f  +  ((V).Tile) ); }
float4  LWCRsqrtUnscaled( FLWCVector4  V) { return rsqrt(V.Offset *  4.76837158e-07f  +  ((V).Tile) ); }
float4  LWCRcpUnscaled( FLWCVector4  V) { return rcp(V.Offset *  4.76837158e-07f  +  ((V).Tile) ); }
float4  LWCSqrtScaled( FLWCVector4  V, float Scale) { return LWCSqrtUnscaled(V) * Scale; }
float4  LWCRsqrtScaled( FLWCVector4  V, float Scale) { return LWCRsqrtUnscaled(V) * Scale; }
float4  LWCRcpScaled( FLWCVector4  V, float Scale) { return LWCRcpUnscaled(V) * Scale; }
float4  LWCSqrt( FLWCVector4  V) { return LWCSqrtScaled(V,  1448.15466f ); }
float4  LWCRsqrt( FLWCVector4  V) { return LWCRsqrtScaled(V,  0.000690533954f ); }
float4  LWCRcp( FLWCVector4  V) { return LWCRcpScaled(V,  4.76837158e-07f ); }
#line 36 "/Engine/Private/LWCOperations.ush"
bool4 LWCGreater( FLWCVector4 Lhs, FLWCVector4 Rhs) { return ( ((Lhs).Tile) - ((Rhs).Tile) ) * 2097152.00f > Rhs.Offset - Lhs.Offset; } bool4 LWCGreater( float4 Lhs, FLWCVector4 Rhs) { return - ((Rhs).Tile) * 2097152.00f > Rhs.Offset - Lhs; } bool4 LWCGreater( FLWCVector4 Lhs, float4 Rhs) { return ((Lhs).Tile) * 2097152.00f > Rhs - Lhs.Offset; }
bool4 LWCGreaterEqual( FLWCVector4 Lhs, FLWCVector4 Rhs) { return ( ((Lhs).Tile) - ((Rhs).Tile) ) * 2097152.00f >= Rhs.Offset - Lhs.Offset; } bool4 LWCGreaterEqual( float4 Lhs, FLWCVector4 Rhs) { return - ((Rhs).Tile) * 2097152.00f >= Rhs.Offset - Lhs; } bool4 LWCGreaterEqual( FLWCVector4 Lhs, float4 Rhs) { return ((Lhs).Tile) * 2097152.00f >= Rhs - Lhs.Offset; }
bool4 LWCLess( FLWCVector4 Lhs, FLWCVector4 Rhs) { return ( ((Lhs).Tile) - ((Rhs).Tile) ) * 2097152.00f < Rhs.Offset - Lhs.Offset; } bool4 LWCLess( float4 Lhs, FLWCVector4 Rhs) { return - ((Rhs).Tile) * 2097152.00f < Rhs.Offset - Lhs; } bool4 LWCLess( FLWCVector4 Lhs, float4 Rhs) { return ((Lhs).Tile) * 2097152.00f < Rhs - Lhs.Offset; }
bool4 LWCLessEqual( FLWCVector4 Lhs, FLWCVector4 Rhs) { return ( ((Lhs).Tile) - ((Rhs).Tile) ) * 2097152.00f <= Rhs.Offset - Lhs.Offset; } bool4 LWCLessEqual( float4 Lhs, FLWCVector4 Rhs) { return - ((Rhs).Tile) * 2097152.00f <= Rhs.Offset - Lhs; } bool4 LWCLessEqual( FLWCVector4 Lhs, float4 Rhs) { return ((Lhs).Tile) * 2097152.00f <= Rhs - Lhs.Offset; }





float4  LWCDdx( FLWCVector4  V) { return ( float4 )0; }
float4  LWCDdy( FLWCVector4  V) { return ( float4 )0; }
#line 52 "/Engine/Private/LWCOperations.ush"
FLWCVector4  LWCAdd( FLWCVector4  Lhs,  FLWCVector4  Rhs) { return  MakeLWCVector4 ( ((Lhs).Tile)  +  ((Rhs).Tile) , Lhs.Offset + Rhs.Offset); }
FLWCVector4  LWCAdd( float4  Lhs,  FLWCVector4  Rhs) { return  MakeLWCVector4 ( ((Rhs).Tile) , Lhs + Rhs.Offset); }
FLWCVector4  LWCAdd( FLWCVector4  Lhs,  float4  Rhs) { return  MakeLWCVector4 ( ((Lhs).Tile) , Lhs.Offset + Rhs); }

FLWCVector4  LWCSubtract( FLWCVector4  Lhs,  FLWCVector4  Rhs) { return  MakeLWCVector4 ( ((Lhs).Tile)  -  ((Rhs).Tile) , Lhs.Offset - Rhs.Offset); }
FLWCVector4  LWCSubtract( float4  Lhs,  FLWCVector4  Rhs) { return  MakeLWCVector4 (- ((Rhs).Tile) , Lhs - Rhs.Offset); }
FLWCVector4  LWCSubtract( FLWCVector4  Lhs,  float4  Rhs) { return  MakeLWCVector4 ( ((Lhs).Tile) , Lhs.Offset - Rhs); }

bool4  LWCEquals( FLWCVector4  Lhs,  FLWCVector4  Rhs)
{
	return ( ((Lhs).Tile)  -  ((Rhs).Tile) ) *  2097152.00f  == Rhs.Offset - Lhs.Offset;
}
bool4  LWCEquals( float4  Lhs,  FLWCVector4  Rhs)
{
	return - ((Rhs).Tile)  *  2097152.00f  == Rhs.Offset - Lhs;
}
bool4  LWCEquals( FLWCVector4  Lhs,  float4  Rhs)
{
	return  ((Lhs).Tile)  *  2097152.00f  == Rhs - Lhs.Offset;
}
bool4  LWCEqualsApprox( FLWCVector4  Lhs,  FLWCVector4  Rhs, float Threshold)
{
	return abs(( ((Lhs).Tile)  -  ((Rhs).Tile) ) *  2097152.00f  + (Lhs.Offset - Rhs.Offset)) < ( float4 )Threshold;
}
bool4  LWCEqualsApprox( float4  Lhs,  FLWCVector4  Rhs, float Threshold)
{
	return abs(- ((Rhs).Tile)  *  2097152.00f  + (Lhs - Rhs.Offset)) < ( float4 )Threshold;
}
bool4  LWCEqualsApprox( FLWCVector4  Lhs,  float4  Rhs, float Threshold)
{
	return abs( ((Lhs).Tile)  *  2097152.00f  + (Lhs.Offset - Rhs)) < ( float4 )Threshold;
}

FLWCVector4  LWCSelect( bool4  S,  FLWCVector4  Lhs,  FLWCVector4  Rhs) { return  MakeLWCVector4 (S ?  ((Lhs).Tile)  :  ((Rhs).Tile) , S ? Lhs.Offset : Rhs.Offset); }
FLWCVector4  LWCSelect( bool4  S,  float4  Lhs,  FLWCVector4  Rhs) { return  MakeLWCVector4 (S ? ( float4 )0 :  ((Rhs).Tile) , S ? Lhs : Rhs.Offset); }
FLWCVector4  LWCSelect( bool4  S,  FLWCVector4  Lhs,  float4  Rhs) { return  MakeLWCVector4 (S ?  ((Lhs).Tile)  : ( float4 )0, S ? Lhs.Offset : Rhs); }

FLWCVector4  LWCNegate( FLWCVector4  V) { return  MakeLWCVector4 (- ((V).Tile) , -V.Offset); }

float4  LWCFrac( FLWCVector4  V)
{
	float4  FracTile = frac( ((V).Tile)  *  2097152.00f );
	return frac(FracTile + V.Offset);
}

FLWCVector4  LWCFloor( FLWCVector4  V) {  FLWCVector4  VN = LWCMakeIntTile(V); return  MakeLWCVector4 ( ((VN).Tile) , floor(VN.Offset)); }
FLWCVector4  LWCCeil( FLWCVector4  V) {  FLWCVector4  VN = LWCMakeIntTile(V); return  MakeLWCVector4 ( ((VN).Tile) , ceil(VN.Offset)); }
FLWCVector4  LWCRound( FLWCVector4  V) {  FLWCVector4  VN = LWCMakeIntTile(V); return  MakeLWCVector4 ( ((VN).Tile) , round(VN.Offset)); }
FLWCVector4  LWCTrunc( FLWCVector4  V) {  FLWCVector4  VN = LWCMakeIntTile(V); return  MakeLWCVector4 ( ((VN).Tile) , trunc(VN.Offset)); }


float4  LWCSign( FLWCVector4  V) { return  float4 (sign(LWCToFloat(V))); }
float4  LWCSaturate( FLWCVector4  V) { return saturate(LWCToFloat(V)); }
float4  LWCClampScalar( FLWCVector4  V, float Low, float High) { return clamp(LWCToFloat(V), Low, High); }

FLWCVector4  LWCMultiply( FLWCVector4  Lhs,  FLWCVector4  Rhs)
{
	return  MakeLWCVector4 ( ((Lhs).Tile)  * ( ((Rhs).Tile)  *  2097152.00f  + Rhs.Offset) +  ((Rhs).Tile)  * Lhs.Offset, Lhs.Offset * Rhs.Offset);
}
FLWCVector4  LWCMultiply( float4  Lhs,  FLWCVector4  Rhs) { return  MakeLWCVector4 ( ((Rhs).Tile)  * Lhs, Lhs * Rhs.Offset); }
FLWCVector4  LWCMultiply( FLWCVector4  Lhs,  float4  Rhs) { return  MakeLWCVector4 ( ((Lhs).Tile)  * Rhs, Lhs.Offset * Rhs); }

FLWCVector4  LWCDivide( FLWCVector4  Lhs,  FLWCVector4  Rhs) { return LWCMultiply(Lhs, LWCRcp(Rhs)); }
FLWCVector4  LWCDivide( FLWCVector4  Lhs,  float4  Rhs) { return LWCMultiply(Lhs, rcp(Rhs)); }
FLWCVector4  LWCDivide( float4  Lhs,  FLWCVector4  Rhs) { return  MakeLWCVector4 (( float4 )0, Lhs * LWCRcp(Rhs)); }


FLWCVector4  LWCLerp( FLWCVector4  Lhs,  FLWCVector4  Rhs,  float4  S)
{
	return  MakeLWCVector4 (lerp( ((Lhs).Tile) ,  ((Rhs).Tile) , S), lerp(Lhs.Offset, Rhs.Offset, S));
}

float4  LWCFmod( FLWCVector4  Lhs,  float4  Rhs)
{
	return LWCToFloat(LWCSubtract(Lhs, LWCMultiply(LWCTrunc(LWCDivide(Lhs, Rhs)), Rhs)));


}
float4  LWCFmodFloor( FLWCVector4  Lhs,  float4  Rhs)
{
	return LWCToFloat(LWCSubtract(Lhs, LWCMultiply(LWCFloor(LWCDivide(Lhs, Rhs)), Rhs)));


}
float4  LWCFmodFloorPI( FLWCVector4  V)
{
	return LWCFmodFloor(V, PI);

}
float4  LWCFmodFloor2PI( FLWCVector4  V)
{
	return LWCFmodFloor(V, 2.0f * PI);

}

float4  LWCSin( FLWCVector4  V) { return sin(LWCFmodFloor2PI(V)); }
float4  LWCCos( FLWCVector4  V) { return cos(LWCFmodFloor2PI(V)); }
float4  LWCTan( FLWCVector4  V) { return tan(LWCFmodFloorPI(V)); }
float4  LWCASin( FLWCVector4  V) { return asin(LWCClampScalar(V, -1.0f, 1.0f)); }
float4  LWCACos( FLWCVector4  V) { return acos(LWCClampScalar(V, -1.0f, 1.0f)); }
float4  LWCATan( FLWCVector4  V) { return atan(LWCClampScalar(V, -0.5f*PI, 0.5f*PI)); }

float4  LWCSmoothStep( FLWCVector4  Lhs,  FLWCVector4  Rhs,  FLWCVector4  S)
{
	float4  t = LWCSaturate(LWCDivide(LWCSubtract(S, Lhs), LWCSubtract(Rhs, Lhs)));
	return t*t*(3.0f - (2.0f*t));
}

FLWCVector4  LWCMin( FLWCVector4  Lhs,  FLWCVector4  Rhs) { return LWCSelect(LWCLess(Lhs, Rhs), Lhs, Rhs); }
FLWCVector4  LWCMin( float4  Lhs,  FLWCVector4  Rhs) { return LWCSelect(LWCLess(Lhs, Rhs), Lhs, Rhs); }
FLWCVector4  LWCMin( FLWCVector4  Lhs,  float4  Rhs) { return LWCSelect(LWCLess(Lhs, Rhs), Lhs, Rhs); }
FLWCVector4  LWCMax( FLWCVector4  Lhs,  FLWCVector4  Rhs) { return LWCSelect(LWCGreater(Lhs, Rhs), Lhs, Rhs); }
FLWCVector4  LWCMax( float4  Lhs,  FLWCVector4  Rhs) { return LWCSelect(LWCGreater(Lhs, Rhs), Lhs, Rhs); }
FLWCVector4  LWCMax( FLWCVector4  Lhs,  float4  Rhs) { return LWCSelect(LWCGreater(Lhs, Rhs), Lhs, Rhs); }

FLWCVector4  LWCAbs( FLWCVector4  V) { return LWCSelect(LWCLess(V, ( float4 )0), LWCNegate(V), V); }

float4  LWCStep( FLWCVector4  Lhs,  FLWCVector4  Rhs) { return LWCGreaterEqual(Rhs, Lhs) ? ( float4 )1.0f : ( float4 )0.0f; }
float4  LWCStep( FLWCVector4  Lhs,  float4  Rhs) { return LWCGreaterEqual(Rhs, Lhs) ? ( float4 )1.0f : ( float4 )0.0f; }
float4  LWCStep( float4  Lhs,  FLWCVector4  Rhs) { return LWCGreaterEqual(Rhs, Lhs) ? ( float4 )1.0f : ( float4 )0.0f; }


FLWCVector4  LWCSquareScaled( FLWCVector4  V)
{
	float4  OffsetScaled = V.Offset *  4.76837158e-07f ;
	return  MakeLWCVector4 ( ((V).Tile)  * ( ((V).Tile)  + OffsetScaled * 2.0f), V.Offset * OffsetScaled);
}
#line 366 "/Engine/Private/LargeWorldCoordinates.ush"





FLWCScalar LWCDot(FLWCScalar Lhs, FLWCScalar Rhs)
{
	return LWCMultiply(Lhs, Rhs);
}
FLWCScalar LWCDot(FLWCScalar Lhs, float Rhs)
{
	return LWCMultiply(Lhs, Rhs);
}

FLWCScalar LWCDot(FLWCVector2 Lhs, FLWCVector2 Rhs)
{
	FLWCScalar X2 = LWCMultiply( LWCGetComponent(Lhs, 0) ,  LWCGetComponent(Rhs, 0) );
	FLWCScalar Y2 = LWCMultiply( LWCGetComponent(Lhs, 1) ,  LWCGetComponent(Rhs, 1) );
	return LWCAdd(X2, Y2);
}
FLWCScalar LWCDot(FLWCVector2 Lhs, float2 Rhs)
{
	FLWCScalar X2 = LWCMultiply( LWCGetComponent(Lhs, 0) , Rhs.x);
	FLWCScalar Y2 = LWCMultiply( LWCGetComponent(Lhs, 1) , Rhs.y);
	return LWCAdd(X2, Y2);
}

FLWCScalar LWCDot(FLWCVector3 Lhs, FLWCVector3 Rhs)
{
	FLWCScalar X2 = LWCMultiply( LWCGetComponent(Lhs, 0) ,  LWCGetComponent(Rhs, 0) );
	FLWCScalar Y2 = LWCMultiply( LWCGetComponent(Lhs, 1) ,  LWCGetComponent(Rhs, 1) );
	FLWCScalar Z2 = LWCMultiply( LWCGetComponent(Lhs, 2) ,  LWCGetComponent(Rhs, 2) );
	return LWCAdd(LWCAdd(X2, Y2), Z2);
}
FLWCScalar LWCDot(FLWCVector3 Lhs, float3 Rhs)
{
	FLWCScalar X2 = LWCMultiply( LWCGetComponent(Lhs, 0) , Rhs.x);
	FLWCScalar Y2 = LWCMultiply( LWCGetComponent(Lhs, 1) , Rhs.y);
	FLWCScalar Z2 = LWCMultiply( LWCGetComponent(Lhs, 2) , Rhs.z);
	return LWCAdd(LWCAdd(X2, Y2), Z2);
}

FLWCScalar LWCDot(FLWCVector4 Lhs, FLWCVector4 Rhs)
{
	FLWCScalar X2 = LWCMultiply( LWCGetComponent(Lhs, 0) ,  LWCGetComponent(Rhs, 0) );
	FLWCScalar Y2 = LWCMultiply( LWCGetComponent(Lhs, 1) ,  LWCGetComponent(Rhs, 1) );
	FLWCScalar Z2 = LWCMultiply( LWCGetComponent(Lhs, 2) ,  LWCGetComponent(Rhs, 2) );
	FLWCScalar W2 = LWCMultiply( LWCGetComponent(Lhs, 3) ,  LWCGetComponent(Rhs, 3) );
	return LWCAdd(LWCAdd(LWCAdd(X2, Y2), Z2), W2);
}
FLWCScalar LWCDot(FLWCVector4 Lhs, float4 Rhs)
{
	FLWCScalar X2 = LWCMultiply( LWCGetComponent(Lhs, 0) , Rhs.x);
	FLWCScalar Y2 = LWCMultiply( LWCGetComponent(Lhs, 1) , Rhs.y);
	FLWCScalar Z2 = LWCMultiply( LWCGetComponent(Lhs, 2) , Rhs.z);
	FLWCScalar W2 = LWCMultiply( LWCGetComponent(Lhs, 3) , Rhs.w);
	return LWCAdd(LWCAdd(LWCAdd(X2, Y2), Z2), W2);
}


FLWCScalar LWCLength2Scaled(FLWCScalar V)
{
	return LWCSquareScaled(V);
}

FLWCScalar LWCLength2Scaled(FLWCVector2 V)
{
	FLWCScalar X2 = LWCSquareScaled( LWCGetComponent(V, 0) );
	FLWCScalar Y2 = LWCSquareScaled( LWCGetComponent(V, 1) );
	return LWCAdd(X2, Y2);
}

FLWCScalar LWCLength2Scaled(FLWCVector3 V)
{
	FLWCScalar X2 = LWCSquareScaled( LWCGetComponent(V, 0) );
	FLWCScalar Y2 = LWCSquareScaled( LWCGetComponent(V, 1) );
	FLWCScalar Z2 = LWCSquareScaled( LWCGetComponent(V, 2) );
	return LWCAdd(LWCAdd(X2, Y2), Z2);
}

FLWCScalar LWCLength2Scaled(FLWCVector4 V)
{
	FLWCScalar X2 = LWCSquareScaled( LWCGetComponent(V, 0) );
	FLWCScalar Y2 = LWCSquareScaled( LWCGetComponent(V, 1) );
	FLWCScalar Z2 = LWCSquareScaled( LWCGetComponent(V, 2) );
	FLWCScalar W2 = LWCSquareScaled( LWCGetComponent(V, 3) );
	return LWCAdd(LWCAdd(LWCAdd(X2, Y2), Z2), W2);
}



FLWCScalar LWCLength(FLWCScalar V) { return MakeLWCScalar(LWCSqrtUnscaled(LWCLength2Scaled(V)), 0.0f); }
FLWCScalar LWCLength(FLWCVector2 V) { return MakeLWCScalar(LWCSqrtUnscaled(LWCLength2Scaled(V)), 0.0f); }
FLWCScalar LWCLength(FLWCVector3 V) { return MakeLWCScalar(LWCSqrtUnscaled(LWCLength2Scaled(V)), 0.0f); }
FLWCScalar LWCLength(FLWCVector4 V) { return MakeLWCScalar(LWCSqrtUnscaled(LWCLength2Scaled(V)), 0.0f); }

float LWCRcpLength(FLWCScalar V) { return LWCRsqrtScaled(LWCLength2Scaled(V),  4.76837158e-07f ); }
float LWCRcpLength(FLWCVector2 V) { return LWCRsqrtScaled(LWCLength2Scaled(V),  4.76837158e-07f ); }
float LWCRcpLength(FLWCVector3 V) { return LWCRsqrtScaled(LWCLength2Scaled(V),  4.76837158e-07f ); }
float LWCRcpLength(FLWCVector4 V) { return LWCRsqrtScaled(LWCLength2Scaled(V),  4.76837158e-07f ); }

float LWCNormalize(FLWCScalar V) { return 1.0f; }
float2 LWCNormalize(FLWCVector2 V) { return LWCToFloat(LWCMultiply(V, LWCRcpLength(V))); }
float3 LWCNormalize(FLWCVector3 V) { return LWCToFloat(LWCMultiply(V, LWCRcpLength(V))); }
float4 LWCNormalize(FLWCVector4 V) { return LWCToFloat(LWCMultiply(V, LWCRcpLength(V))); }
#line 197 "/Engine/Private/Common.ush"
#line 198 "/Engine/Private/Common.ush"
#line 1 "InstancedStereo.ush"
#line 10 "/Engine/Private/InstancedStereo.ush"
#line 1 "/Engine/Generated/UniformBuffers/View.ush"
#line 11 "/Engine/Private/InstancedStereo.ush"
#line 1 "/Engine/Generated/UniformBuffers/InstancedView.ush"
#line 12 "/Engine/Private/InstancedStereo.ush"
#line 15 "/Engine/Private/InstancedStereo.ush"
#line 1 "/Engine/Generated/GeneratedInstancedStereo.ush"
struct ViewState
{
	float4x4 TranslatedWorldToClip;
	float4x4 RelativeWorldToClip;
	float4x4 ClipToRelativeWorld;
	float4x4 TranslatedWorldToView;
	float4x4 ViewToTranslatedWorld;
	float4x4 TranslatedWorldToCameraView;
	float4x4 CameraViewToTranslatedWorld;
	float4x4 ViewToClip;
	float4x4 ViewToClipNoAA;
	float4x4 ClipToView;
	float4x4 ClipToTranslatedWorld;
	float4x4 SVPositionToTranslatedWorld;
	float4x4 ScreenToRelativeWorld;
	float4x4 ScreenToTranslatedWorld;
	float4x4 MobileMultiviewShadowTransform;
	float3 ViewTilePosition;
	float3 MatrixTilePosition;
	float3 ViewForward;
	float3 ViewUp;
	float3 ViewRight;
	float3 HMDViewNoRollUp;
	float3 HMDViewNoRollRight;
	float4 InvDeviceZToWorldZTransform;
	float4 ScreenPositionScaleBias;
	float3 RelativeWorldCameraOrigin;
	float3 TranslatedWorldCameraOrigin;
	float3 RelativeWorldViewOrigin;
	float3 RelativePreViewTranslation;
	float4x4 PrevViewToClip;
	float4x4 PrevClipToView;
	float4x4 PrevTranslatedWorldToClip;
	float4x4 PrevTranslatedWorldToView;
	float4x4 PrevViewToTranslatedWorld;
	float4x4 PrevTranslatedWorldToCameraView;
	float4x4 PrevCameraViewToTranslatedWorld;
	float3 PrevTranslatedWorldCameraOrigin;
	float3 PrevRelativeWorldCameraOrigin;
	float3 PrevRelativeWorldViewOrigin;
	float3 RelativePrevPreViewTranslation;
	float4x4 PrevClipToRelativeWorld;
	float4x4 PrevScreenToTranslatedWorld;
	float4x4 ClipToPrevClip;
	float4x4 ClipToPrevClipWithAA;
	float4 TemporalAAJitter;
	float4 GlobalClippingPlane;
	float2 FieldOfViewWideAngles;
	float2 PrevFieldOfViewWideAngles;
	float4 ViewRectMin;
	float4 ViewSizeAndInvSize;
	float4 LightProbeSizeRatioAndInvSizeRatio;
	float4 BufferSizeAndInvSize;
	float4 BufferBilinearUVMinMax;
	float4 ScreenToViewSpace;
	int NumSceneColorMSAASamples;
	float PreExposure;
	float OneOverPreExposure;
	float4 DiffuseOverrideParameter;
	float4 SpecularOverrideParameter;
	float4 NormalOverrideParameter;
	float2 RoughnessOverrideParameter;
	float PrevFrameGameTime;
	float PrevFrameRealTime;
	float OutOfBoundsMask;
	float3 WorldCameraMovementSinceLastFrame;
	float CullingSign;
	float NearPlane;
	float GameTime;
	float RealTime;
	float DeltaTime;
	float MaterialTextureMipBias;
	float MaterialTextureDerivativeMultiply;
	uint Random;
	uint FrameNumber;
	uint StateFrameIndexMod8;
	uint StateFrameIndex;
	uint StateRawFrameIndex;
	float4 AntiAliasingSampleParams;
	uint DebugViewModeMask;
	uint DebugInput0;
	float CameraCut;
	float UnlitViewmodeMask;
	float4 DirectionalLightColor;
	float3 DirectionalLightDirection;
	float4 TranslucencyLightingVolumeMin[2];
	float4 TranslucencyLightingVolumeInvSize[2];
	float4 TemporalAAParams;
	float4 CircleDOFParams;
	uint ForceDrawAllVelocities;
	float DepthOfFieldSensorWidth;
	float DepthOfFieldFocalDistance;
	float DepthOfFieldScale;
	float DepthOfFieldFocalLength;
	float DepthOfFieldFocalRegion;
	float DepthOfFieldNearTransitionRegion;
	float DepthOfFieldFarTransitionRegion;
	float MotionBlurNormalizedToPixel;
	float GeneralPurposeTweak;
	float GeneralPurposeTweak2;
	float DemosaicVposOffset;
	float DecalDepthBias;
	float3 IndirectLightingColorScale;
	float3 PrecomputedIndirectLightingColorScale;
	float3 PrecomputedIndirectSpecularColorScale;
	float4 AtmosphereLightDirection[2];
	float4 AtmosphereLightIlluminanceOnGroundPostTransmittance[2];
	float4 AtmosphereLightIlluminanceOuterSpace[2];
	float4 AtmosphereLightDiscLuminance[2];
	float4 AtmosphereLightDiscCosHalfApexAngle[2];
	float4 SkyViewLutSizeAndInvSize;
	float3 SkyCameraTranslatedWorldOrigin;
	float4 SkyPlanetTranslatedWorldCenterAndViewHeight;
	float4x4 SkyViewLutReferential;
	float4 SkyAtmosphereSkyLuminanceFactor;
	float SkyAtmospherePresentInScene;
	float SkyAtmosphereHeightFogContribution;
	float SkyAtmosphereBottomRadiusKm;
	float SkyAtmosphereTopRadiusKm;
	float4 SkyAtmosphereCameraAerialPerspectiveVolumeSizeAndInvSize;
	float SkyAtmosphereAerialPerspectiveStartDepthKm;
	float SkyAtmosphereCameraAerialPerspectiveVolumeDepthResolution;
	float SkyAtmosphereCameraAerialPerspectiveVolumeDepthResolutionInv;
	float SkyAtmosphereCameraAerialPerspectiveVolumeDepthSliceLengthKm;
	float SkyAtmosphereCameraAerialPerspectiveVolumeDepthSliceLengthKmInv;
	float SkyAtmosphereApplyCameraAerialPerspectiveVolume;
	float3 NormalCurvatureToRoughnessScaleBias;
	float RenderingReflectionCaptureMask;
	float RealTimeReflectionCapture;
	float RealTimeReflectionCapturePreExposure;
	float4 AmbientCubemapTint;
	float AmbientCubemapIntensity;
	float SkyLightApplyPrecomputedBentNormalShadowingFlag;
	float SkyLightAffectReflectionFlag;
	float SkyLightAffectGlobalIlluminationFlag;
	float4 SkyLightColor;
	float4 MobileSkyIrradianceEnvironmentMap[7];
	float MobilePreviewMode;
	float HMDEyePaddingOffset;
	float ReflectionCubemapMaxMip;
	float ShowDecalsMask;
	uint DistanceFieldAOSpecularOcclusionMode;
	float IndirectCapsuleSelfShadowingIntensity;
	float3 ReflectionEnvironmentRoughnessMixingScaleBiasAndLargestWeight;
	int StereoPassIndex;
	float4 GlobalVolumeCenterAndExtent[6];
	float4 GlobalVolumeWorldToUVAddAndMul[6];
	float4 GlobalDistanceFieldMipWorldToUVScale[6];
	float4 GlobalDistanceFieldMipWorldToUVBias[6];
	float GlobalDistanceFieldMipFactor;
	float GlobalDistanceFieldMipTransition;
	int GlobalDistanceFieldClipmapSizeInPages;
	float3 GlobalDistanceFieldInvPageAtlasSize;
	float3 GlobalDistanceFieldInvCoverageAtlasSize;
	float GlobalVolumeDimension;
	float GlobalVolumeTexelSize;
	float MaxGlobalDFAOConeDistance;
	uint NumGlobalSDFClipmaps;
	float FullyCoveredExpandSurfaceScale;
	float UncoveredExpandSurfaceScale;
	float UncoveredMinStepScale;
	int2 CursorPosition;
	float bCheckerboardSubsurfaceProfileRendering;
	float3 VolumetricFogInvGridSize;
	float3 VolumetricFogGridZParams;
	float2 VolumetricFogSVPosToVolumeUV;
	float VolumetricFogMaxDistance;
	float3 VolumetricLightmapWorldToUVScale;
	float3 VolumetricLightmapWorldToUVAdd;
	float3 VolumetricLightmapIndirectionTextureSize;
	float VolumetricLightmapBrickSize;
	float3 VolumetricLightmapBrickTexelSize;
	float StereoIPD;
	float IndirectLightingCacheShowFlag;
	float EyeToPixelSpreadAngle;
	float4 XRPassthroughCameraUVs[2];
	float GlobalVirtualTextureMipBias;
	uint VirtualTextureFeedbackShift;
	uint VirtualTextureFeedbackMask;
	uint VirtualTextureFeedbackStride;
	uint VirtualTextureFeedbackJitterOffset;
	uint VirtualTextureFeedbackSampleOffset;
	float4 RuntimeVirtualTextureMipLevel;
	float2 RuntimeVirtualTexturePackHeight;
	float4 RuntimeVirtualTextureDebugParams;
	float OverrideLandscapeLOD;
	int FarShadowStaticMeshLODBias;
	float MinRoughness;
	float4 HairRenderInfo;
	uint EnableSkyLight;
	uint HairRenderInfoBits;
	uint HairComponents;
	float bSubsurfacePostprocessEnabled;
	float4 SSProfilesTextureSizeAndInvSize;
	float4 SSProfilesPreIntegratedTextureSizeAndInvSize;
	float3 PhysicsFieldClipmapCenter;
	float PhysicsFieldClipmapDistance;
	int PhysicsFieldClipmapResolution;
	int PhysicsFieldClipmapExponent;
	int PhysicsFieldClipmapCount;
	int PhysicsFieldTargetCount;
	int4 PhysicsFieldTargets[32];
	uint InstanceSceneDataSOAStride;
	uint GPUSceneViewId;
	FLWCInverseMatrix WorldToClip;
	FLWCMatrix ClipToWorld;
	FLWCMatrix ScreenToWorld;
	FLWCMatrix PrevClipToWorld;
	FLWCVector3 WorldCameraOrigin;
	FLWCVector3 WorldViewOrigin;
	FLWCVector3 PrevWorldCameraOrigin;
	FLWCVector3 PrevWorldViewOrigin;
	FLWCVector3 PreViewTranslation;
	FLWCVector3 PrevPreViewTranslation;
};
	void FinalizeViewState(inout ViewState InOutView);
ViewState GetPrimaryView()
{
	ViewState Result;
	Result.TranslatedWorldToClip = View_TranslatedWorldToClip;
	Result.RelativeWorldToClip = View_RelativeWorldToClip;
	Result.ClipToRelativeWorld = View_ClipToRelativeWorld;
	Result.TranslatedWorldToView = View_TranslatedWorldToView;
	Result.ViewToTranslatedWorld = View_ViewToTranslatedWorld;
	Result.TranslatedWorldToCameraView = View_TranslatedWorldToCameraView;
	Result.CameraViewToTranslatedWorld = View_CameraViewToTranslatedWorld;
	Result.ViewToClip = View_ViewToClip;
	Result.ViewToClipNoAA = View_ViewToClipNoAA;
	Result.ClipToView = View_ClipToView;
	Result.ClipToTranslatedWorld = View_ClipToTranslatedWorld;
	Result.SVPositionToTranslatedWorld = View_SVPositionToTranslatedWorld;
	Result.ScreenToRelativeWorld = View_ScreenToRelativeWorld;
	Result.ScreenToTranslatedWorld = View_ScreenToTranslatedWorld;
	Result.MobileMultiviewShadowTransform = View_MobileMultiviewShadowTransform;
	Result.ViewTilePosition = View_ViewTilePosition;
	Result.MatrixTilePosition = View_MatrixTilePosition;
	Result.ViewForward = View_ViewForward;
	Result.ViewUp = View_ViewUp;
	Result.ViewRight = View_ViewRight;
	Result.HMDViewNoRollUp = View_HMDViewNoRollUp;
	Result.HMDViewNoRollRight = View_HMDViewNoRollRight;
	Result.InvDeviceZToWorldZTransform = View_InvDeviceZToWorldZTransform;
	Result.ScreenPositionScaleBias = View_ScreenPositionScaleBias;
	Result.RelativeWorldCameraOrigin = View_RelativeWorldCameraOrigin;
	Result.TranslatedWorldCameraOrigin = View_TranslatedWorldCameraOrigin;
	Result.RelativeWorldViewOrigin = View_RelativeWorldViewOrigin;
	Result.RelativePreViewTranslation = View_RelativePreViewTranslation;
	Result.PrevViewToClip = View_PrevViewToClip;
	Result.PrevClipToView = View_PrevClipToView;
	Result.PrevTranslatedWorldToClip = View_PrevTranslatedWorldToClip;
	Result.PrevTranslatedWorldToView = View_PrevTranslatedWorldToView;
	Result.PrevViewToTranslatedWorld = View_PrevViewToTranslatedWorld;
	Result.PrevTranslatedWorldToCameraView = View_PrevTranslatedWorldToCameraView;
	Result.PrevCameraViewToTranslatedWorld = View_PrevCameraViewToTranslatedWorld;
	Result.PrevTranslatedWorldCameraOrigin = View_PrevTranslatedWorldCameraOrigin;
	Result.PrevRelativeWorldCameraOrigin = View_PrevRelativeWorldCameraOrigin;
	Result.PrevRelativeWorldViewOrigin = View_PrevRelativeWorldViewOrigin;
	Result.RelativePrevPreViewTranslation = View_RelativePrevPreViewTranslation;
	Result.PrevClipToRelativeWorld = View_PrevClipToRelativeWorld;
	Result.PrevScreenToTranslatedWorld = View_PrevScreenToTranslatedWorld;
	Result.ClipToPrevClip = View_ClipToPrevClip;
	Result.ClipToPrevClipWithAA = View_ClipToPrevClipWithAA;
	Result.TemporalAAJitter = View_TemporalAAJitter;
	Result.GlobalClippingPlane = View_GlobalClippingPlane;
	Result.FieldOfViewWideAngles = View_FieldOfViewWideAngles;
	Result.PrevFieldOfViewWideAngles = View_PrevFieldOfViewWideAngles;
	Result.ViewRectMin = View_ViewRectMin;
	Result.ViewSizeAndInvSize = View_ViewSizeAndInvSize;
	Result.LightProbeSizeRatioAndInvSizeRatio = View_LightProbeSizeRatioAndInvSizeRatio;
	Result.BufferSizeAndInvSize = View_BufferSizeAndInvSize;
	Result.BufferBilinearUVMinMax = View_BufferBilinearUVMinMax;
	Result.ScreenToViewSpace = View_ScreenToViewSpace;
	Result.NumSceneColorMSAASamples = View_NumSceneColorMSAASamples;
	Result.PreExposure = View_PreExposure;
	Result.OneOverPreExposure = View_OneOverPreExposure;
	Result.DiffuseOverrideParameter = View_DiffuseOverrideParameter;
	Result.SpecularOverrideParameter = View_SpecularOverrideParameter;
	Result.NormalOverrideParameter = View_NormalOverrideParameter;
	Result.RoughnessOverrideParameter = View_RoughnessOverrideParameter;
	Result.PrevFrameGameTime = View_PrevFrameGameTime;
	Result.PrevFrameRealTime = View_PrevFrameRealTime;
	Result.OutOfBoundsMask = View_OutOfBoundsMask;
	Result.WorldCameraMovementSinceLastFrame = View_WorldCameraMovementSinceLastFrame;
	Result.CullingSign = View_CullingSign;
	Result.NearPlane = View_NearPlane;
	Result.GameTime = View_GameTime;
	Result.RealTime = View_RealTime;
	Result.DeltaTime = View_DeltaTime;
	Result.MaterialTextureMipBias = View_MaterialTextureMipBias;
	Result.MaterialTextureDerivativeMultiply = View_MaterialTextureDerivativeMultiply;
	Result.Random = View_Random;
	Result.FrameNumber = View_FrameNumber;
	Result.StateFrameIndexMod8 = View_StateFrameIndexMod8;
	Result.StateFrameIndex = View_StateFrameIndex;
	Result.StateRawFrameIndex = View_StateRawFrameIndex;
	Result.AntiAliasingSampleParams = View_AntiAliasingSampleParams;
	Result.DebugViewModeMask = View_DebugViewModeMask;
	Result.DebugInput0 = View_DebugInput0;
	Result.CameraCut = View_CameraCut;
	Result.UnlitViewmodeMask = View_UnlitViewmodeMask;
	Result.DirectionalLightColor = View_DirectionalLightColor;
	Result.DirectionalLightDirection = View_DirectionalLightDirection;
	Result.TranslucencyLightingVolumeMin = View_TranslucencyLightingVolumeMin;
	Result.TranslucencyLightingVolumeInvSize = View_TranslucencyLightingVolumeInvSize;
	Result.TemporalAAParams = View_TemporalAAParams;
	Result.CircleDOFParams = View_CircleDOFParams;
	Result.ForceDrawAllVelocities = View_ForceDrawAllVelocities;
	Result.DepthOfFieldSensorWidth = View_DepthOfFieldSensorWidth;
	Result.DepthOfFieldFocalDistance = View_DepthOfFieldFocalDistance;
	Result.DepthOfFieldScale = View_DepthOfFieldScale;
	Result.DepthOfFieldFocalLength = View_DepthOfFieldFocalLength;
	Result.DepthOfFieldFocalRegion = View_DepthOfFieldFocalRegion;
	Result.DepthOfFieldNearTransitionRegion = View_DepthOfFieldNearTransitionRegion;
	Result.DepthOfFieldFarTransitionRegion = View_DepthOfFieldFarTransitionRegion;
	Result.MotionBlurNormalizedToPixel = View_MotionBlurNormalizedToPixel;
	Result.GeneralPurposeTweak = View_GeneralPurposeTweak;
	Result.GeneralPurposeTweak2 = View_GeneralPurposeTweak2;
	Result.DemosaicVposOffset = View_DemosaicVposOffset;
	Result.DecalDepthBias = View_DecalDepthBias;
	Result.IndirectLightingColorScale = View_IndirectLightingColorScale;
	Result.PrecomputedIndirectLightingColorScale = View_PrecomputedIndirectLightingColorScale;
	Result.PrecomputedIndirectSpecularColorScale = View_PrecomputedIndirectSpecularColorScale;
	Result.AtmosphereLightDirection = View_AtmosphereLightDirection;
	Result.AtmosphereLightIlluminanceOnGroundPostTransmittance = View_AtmosphereLightIlluminanceOnGroundPostTransmittance;
	Result.AtmosphereLightIlluminanceOuterSpace = View_AtmosphereLightIlluminanceOuterSpace;
	Result.AtmosphereLightDiscLuminance = View_AtmosphereLightDiscLuminance;
	Result.AtmosphereLightDiscCosHalfApexAngle = View_AtmosphereLightDiscCosHalfApexAngle;
	Result.SkyViewLutSizeAndInvSize = View_SkyViewLutSizeAndInvSize;
	Result.SkyCameraTranslatedWorldOrigin = View_SkyCameraTranslatedWorldOrigin;
	Result.SkyPlanetTranslatedWorldCenterAndViewHeight = View_SkyPlanetTranslatedWorldCenterAndViewHeight;
	Result.SkyViewLutReferential = View_SkyViewLutReferential;
	Result.SkyAtmosphereSkyLuminanceFactor = View_SkyAtmosphereSkyLuminanceFactor;
	Result.SkyAtmospherePresentInScene = View_SkyAtmospherePresentInScene;
	Result.SkyAtmosphereHeightFogContribution = View_SkyAtmosphereHeightFogContribution;
	Result.SkyAtmosphereBottomRadiusKm = View_SkyAtmosphereBottomRadiusKm;
	Result.SkyAtmosphereTopRadiusKm = View_SkyAtmosphereTopRadiusKm;
	Result.SkyAtmosphereCameraAerialPerspectiveVolumeSizeAndInvSize = View_SkyAtmosphereCameraAerialPerspectiveVolumeSizeAndInvSize;
	Result.SkyAtmosphereAerialPerspectiveStartDepthKm = View_SkyAtmosphereAerialPerspectiveStartDepthKm;
	Result.SkyAtmosphereCameraAerialPerspectiveVolumeDepthResolution = View_SkyAtmosphereCameraAerialPerspectiveVolumeDepthResolution;
	Result.SkyAtmosphereCameraAerialPerspectiveVolumeDepthResolutionInv = View_SkyAtmosphereCameraAerialPerspectiveVolumeDepthResolutionInv;
	Result.SkyAtmosphereCameraAerialPerspectiveVolumeDepthSliceLengthKm = View_SkyAtmosphereCameraAerialPerspectiveVolumeDepthSliceLengthKm;
	Result.SkyAtmosphereCameraAerialPerspectiveVolumeDepthSliceLengthKmInv = View_SkyAtmosphereCameraAerialPerspectiveVolumeDepthSliceLengthKmInv;
	Result.SkyAtmosphereApplyCameraAerialPerspectiveVolume = View_SkyAtmosphereApplyCameraAerialPerspectiveVolume;
	Result.NormalCurvatureToRoughnessScaleBias = View_NormalCurvatureToRoughnessScaleBias;
	Result.RenderingReflectionCaptureMask = View_RenderingReflectionCaptureMask;
	Result.RealTimeReflectionCapture = View_RealTimeReflectionCapture;
	Result.RealTimeReflectionCapturePreExposure = View_RealTimeReflectionCapturePreExposure;
	Result.AmbientCubemapTint = View_AmbientCubemapTint;
	Result.AmbientCubemapIntensity = View_AmbientCubemapIntensity;
	Result.SkyLightApplyPrecomputedBentNormalShadowingFlag = View_SkyLightApplyPrecomputedBentNormalShadowingFlag;
	Result.SkyLightAffectReflectionFlag = View_SkyLightAffectReflectionFlag;
	Result.SkyLightAffectGlobalIlluminationFlag = View_SkyLightAffectGlobalIlluminationFlag;
	Result.SkyLightColor = View_SkyLightColor;
	Result.MobileSkyIrradianceEnvironmentMap = View_MobileSkyIrradianceEnvironmentMap;
	Result.MobilePreviewMode = View_MobilePreviewMode;
	Result.HMDEyePaddingOffset = View_HMDEyePaddingOffset;
	Result.ReflectionCubemapMaxMip = View_ReflectionCubemapMaxMip;
	Result.ShowDecalsMask = View_ShowDecalsMask;
	Result.DistanceFieldAOSpecularOcclusionMode = View_DistanceFieldAOSpecularOcclusionMode;
	Result.IndirectCapsuleSelfShadowingIntensity = View_IndirectCapsuleSelfShadowingIntensity;
	Result.ReflectionEnvironmentRoughnessMixingScaleBiasAndLargestWeight = View_ReflectionEnvironmentRoughnessMixingScaleBiasAndLargestWeight;
	Result.StereoPassIndex = View_StereoPassIndex;
	Result.GlobalVolumeCenterAndExtent = View_GlobalVolumeCenterAndExtent;
	Result.GlobalVolumeWorldToUVAddAndMul = View_GlobalVolumeWorldToUVAddAndMul;
	Result.GlobalDistanceFieldMipWorldToUVScale = View_GlobalDistanceFieldMipWorldToUVScale;
	Result.GlobalDistanceFieldMipWorldToUVBias = View_GlobalDistanceFieldMipWorldToUVBias;
	Result.GlobalDistanceFieldMipFactor = View_GlobalDistanceFieldMipFactor;
	Result.GlobalDistanceFieldMipTransition = View_GlobalDistanceFieldMipTransition;
	Result.GlobalDistanceFieldClipmapSizeInPages = View_GlobalDistanceFieldClipmapSizeInPages;
	Result.GlobalDistanceFieldInvPageAtlasSize = View_GlobalDistanceFieldInvPageAtlasSize;
	Result.GlobalDistanceFieldInvCoverageAtlasSize = View_GlobalDistanceFieldInvCoverageAtlasSize;
	Result.GlobalVolumeDimension = View_GlobalVolumeDimension;
	Result.GlobalVolumeTexelSize = View_GlobalVolumeTexelSize;
	Result.MaxGlobalDFAOConeDistance = View_MaxGlobalDFAOConeDistance;
	Result.NumGlobalSDFClipmaps = View_NumGlobalSDFClipmaps;
	Result.FullyCoveredExpandSurfaceScale = View_FullyCoveredExpandSurfaceScale;
	Result.UncoveredExpandSurfaceScale = View_UncoveredExpandSurfaceScale;
	Result.UncoveredMinStepScale = View_UncoveredMinStepScale;
	Result.CursorPosition = View_CursorPosition;
	Result.bCheckerboardSubsurfaceProfileRendering = View_bCheckerboardSubsurfaceProfileRendering;
	Result.VolumetricFogInvGridSize = View_VolumetricFogInvGridSize;
	Result.VolumetricFogGridZParams = View_VolumetricFogGridZParams;
	Result.VolumetricFogSVPosToVolumeUV = View_VolumetricFogSVPosToVolumeUV;
	Result.VolumetricFogMaxDistance = View_VolumetricFogMaxDistance;
	Result.VolumetricLightmapWorldToUVScale = View_VolumetricLightmapWorldToUVScale;
	Result.VolumetricLightmapWorldToUVAdd = View_VolumetricLightmapWorldToUVAdd;
	Result.VolumetricLightmapIndirectionTextureSize = View_VolumetricLightmapIndirectionTextureSize;
	Result.VolumetricLightmapBrickSize = View_VolumetricLightmapBrickSize;
	Result.VolumetricLightmapBrickTexelSize = View_VolumetricLightmapBrickTexelSize;
	Result.StereoIPD = View_StereoIPD;
	Result.IndirectLightingCacheShowFlag = View_IndirectLightingCacheShowFlag;
	Result.EyeToPixelSpreadAngle = View_EyeToPixelSpreadAngle;
	Result.XRPassthroughCameraUVs = View_XRPassthroughCameraUVs;
	Result.GlobalVirtualTextureMipBias = View_GlobalVirtualTextureMipBias;
	Result.VirtualTextureFeedbackShift = View_VirtualTextureFeedbackShift;
	Result.VirtualTextureFeedbackMask = View_VirtualTextureFeedbackMask;
	Result.VirtualTextureFeedbackStride = View_VirtualTextureFeedbackStride;
	Result.VirtualTextureFeedbackJitterOffset = View_VirtualTextureFeedbackJitterOffset;
	Result.VirtualTextureFeedbackSampleOffset = View_VirtualTextureFeedbackSampleOffset;
	Result.RuntimeVirtualTextureMipLevel = View_RuntimeVirtualTextureMipLevel;
	Result.RuntimeVirtualTexturePackHeight = View_RuntimeVirtualTexturePackHeight;
	Result.RuntimeVirtualTextureDebugParams = View_RuntimeVirtualTextureDebugParams;
	Result.OverrideLandscapeLOD = View_OverrideLandscapeLOD;
	Result.FarShadowStaticMeshLODBias = View_FarShadowStaticMeshLODBias;
	Result.MinRoughness = View_MinRoughness;
	Result.HairRenderInfo = View_HairRenderInfo;
	Result.EnableSkyLight = View_EnableSkyLight;
	Result.HairRenderInfoBits = View_HairRenderInfoBits;
	Result.HairComponents = View_HairComponents;
	Result.bSubsurfacePostprocessEnabled = View_bSubsurfacePostprocessEnabled;
	Result.SSProfilesTextureSizeAndInvSize = View_SSProfilesTextureSizeAndInvSize;
	Result.SSProfilesPreIntegratedTextureSizeAndInvSize = View_SSProfilesPreIntegratedTextureSizeAndInvSize;
	Result.PhysicsFieldClipmapCenter = View_PhysicsFieldClipmapCenter;
	Result.PhysicsFieldClipmapDistance = View_PhysicsFieldClipmapDistance;
	Result.PhysicsFieldClipmapResolution = View_PhysicsFieldClipmapResolution;
	Result.PhysicsFieldClipmapExponent = View_PhysicsFieldClipmapExponent;
	Result.PhysicsFieldClipmapCount = View_PhysicsFieldClipmapCount;
	Result.PhysicsFieldTargetCount = View_PhysicsFieldTargetCount;
	Result.PhysicsFieldTargets = View_PhysicsFieldTargets;
	Result.InstanceSceneDataSOAStride = View_InstanceSceneDataSOAStride;
	Result.GPUSceneViewId = View_GPUSceneViewId;
	FinalizeViewState(Result);
	return Result;
}
ViewState GetInstancedView()
{
	ViewState Result;
	Result.TranslatedWorldToClip = InstancedView_TranslatedWorldToClip;
	Result.RelativeWorldToClip = InstancedView_RelativeWorldToClip;
	Result.ClipToRelativeWorld = InstancedView_ClipToRelativeWorld;
	Result.TranslatedWorldToView = InstancedView_TranslatedWorldToView;
	Result.ViewToTranslatedWorld = InstancedView_ViewToTranslatedWorld;
	Result.TranslatedWorldToCameraView = InstancedView_TranslatedWorldToCameraView;
	Result.CameraViewToTranslatedWorld = InstancedView_CameraViewToTranslatedWorld;
	Result.ViewToClip = InstancedView_ViewToClip;
	Result.ViewToClipNoAA = InstancedView_ViewToClipNoAA;
	Result.ClipToView = InstancedView_ClipToView;
	Result.ClipToTranslatedWorld = InstancedView_ClipToTranslatedWorld;
	Result.SVPositionToTranslatedWorld = InstancedView_SVPositionToTranslatedWorld;
	Result.ScreenToRelativeWorld = InstancedView_ScreenToRelativeWorld;
	Result.ScreenToTranslatedWorld = InstancedView_ScreenToTranslatedWorld;
	Result.MobileMultiviewShadowTransform = InstancedView_MobileMultiviewShadowTransform;
	Result.ViewTilePosition = InstancedView_ViewTilePosition;
	Result.MatrixTilePosition = InstancedView_MatrixTilePosition;
	Result.ViewForward = InstancedView_ViewForward;
	Result.ViewUp = InstancedView_ViewUp;
	Result.ViewRight = InstancedView_ViewRight;
	Result.HMDViewNoRollUp = InstancedView_HMDViewNoRollUp;
	Result.HMDViewNoRollRight = InstancedView_HMDViewNoRollRight;
	Result.InvDeviceZToWorldZTransform = InstancedView_InvDeviceZToWorldZTransform;
	Result.ScreenPositionScaleBias = InstancedView_ScreenPositionScaleBias;
	Result.RelativeWorldCameraOrigin = InstancedView_RelativeWorldCameraOrigin;
	Result.TranslatedWorldCameraOrigin = InstancedView_TranslatedWorldCameraOrigin;
	Result.RelativeWorldViewOrigin = InstancedView_RelativeWorldViewOrigin;
	Result.RelativePreViewTranslation = InstancedView_RelativePreViewTranslation;
	Result.PrevViewToClip = InstancedView_PrevViewToClip;
	Result.PrevClipToView = InstancedView_PrevClipToView;
	Result.PrevTranslatedWorldToClip = InstancedView_PrevTranslatedWorldToClip;
	Result.PrevTranslatedWorldToView = InstancedView_PrevTranslatedWorldToView;
	Result.PrevViewToTranslatedWorld = InstancedView_PrevViewToTranslatedWorld;
	Result.PrevTranslatedWorldToCameraView = InstancedView_PrevTranslatedWorldToCameraView;
	Result.PrevCameraViewToTranslatedWorld = InstancedView_PrevCameraViewToTranslatedWorld;
	Result.PrevTranslatedWorldCameraOrigin = InstancedView_PrevTranslatedWorldCameraOrigin;
	Result.PrevRelativeWorldCameraOrigin = InstancedView_PrevRelativeWorldCameraOrigin;
	Result.PrevRelativeWorldViewOrigin = InstancedView_PrevRelativeWorldViewOrigin;
	Result.RelativePrevPreViewTranslation = InstancedView_RelativePrevPreViewTranslation;
	Result.PrevClipToRelativeWorld = InstancedView_PrevClipToRelativeWorld;
	Result.PrevScreenToTranslatedWorld = InstancedView_PrevScreenToTranslatedWorld;
	Result.ClipToPrevClip = InstancedView_ClipToPrevClip;
	Result.ClipToPrevClipWithAA = InstancedView_ClipToPrevClipWithAA;
	Result.TemporalAAJitter = InstancedView_TemporalAAJitter;
	Result.GlobalClippingPlane = InstancedView_GlobalClippingPlane;
	Result.FieldOfViewWideAngles = InstancedView_FieldOfViewWideAngles;
	Result.PrevFieldOfViewWideAngles = InstancedView_PrevFieldOfViewWideAngles;
	Result.ViewRectMin = InstancedView_ViewRectMin;
	Result.ViewSizeAndInvSize = InstancedView_ViewSizeAndInvSize;
	Result.LightProbeSizeRatioAndInvSizeRatio = InstancedView_LightProbeSizeRatioAndInvSizeRatio;
	Result.BufferSizeAndInvSize = InstancedView_BufferSizeAndInvSize;
	Result.BufferBilinearUVMinMax = InstancedView_BufferBilinearUVMinMax;
	Result.ScreenToViewSpace = InstancedView_ScreenToViewSpace;
	Result.NumSceneColorMSAASamples = InstancedView_NumSceneColorMSAASamples;
	Result.PreExposure = InstancedView_PreExposure;
	Result.OneOverPreExposure = InstancedView_OneOverPreExposure;
	Result.DiffuseOverrideParameter = InstancedView_DiffuseOverrideParameter;
	Result.SpecularOverrideParameter = InstancedView_SpecularOverrideParameter;
	Result.NormalOverrideParameter = InstancedView_NormalOverrideParameter;
	Result.RoughnessOverrideParameter = InstancedView_RoughnessOverrideParameter;
	Result.PrevFrameGameTime = InstancedView_PrevFrameGameTime;
	Result.PrevFrameRealTime = InstancedView_PrevFrameRealTime;
	Result.OutOfBoundsMask = InstancedView_OutOfBoundsMask;
	Result.WorldCameraMovementSinceLastFrame = InstancedView_WorldCameraMovementSinceLastFrame;
	Result.CullingSign = InstancedView_CullingSign;
	Result.NearPlane = InstancedView_NearPlane;
	Result.GameTime = InstancedView_GameTime;
	Result.RealTime = InstancedView_RealTime;
	Result.DeltaTime = InstancedView_DeltaTime;
	Result.MaterialTextureMipBias = InstancedView_MaterialTextureMipBias;
	Result.MaterialTextureDerivativeMultiply = InstancedView_MaterialTextureDerivativeMultiply;
	Result.Random = InstancedView_Random;
	Result.FrameNumber = InstancedView_FrameNumber;
	Result.StateFrameIndexMod8 = InstancedView_StateFrameIndexMod8;
	Result.StateFrameIndex = InstancedView_StateFrameIndex;
	Result.StateRawFrameIndex = InstancedView_StateRawFrameIndex;
	Result.AntiAliasingSampleParams = InstancedView_AntiAliasingSampleParams;
	Result.DebugViewModeMask = InstancedView_DebugViewModeMask;
	Result.DebugInput0 = InstancedView_DebugInput0;
	Result.CameraCut = InstancedView_CameraCut;
	Result.UnlitViewmodeMask = InstancedView_UnlitViewmodeMask;
	Result.DirectionalLightColor = InstancedView_DirectionalLightColor;
	Result.DirectionalLightDirection = InstancedView_DirectionalLightDirection;
	Result.TranslucencyLightingVolumeMin = InstancedView_TranslucencyLightingVolumeMin;
	Result.TranslucencyLightingVolumeInvSize = InstancedView_TranslucencyLightingVolumeInvSize;
	Result.TemporalAAParams = InstancedView_TemporalAAParams;
	Result.CircleDOFParams = InstancedView_CircleDOFParams;
	Result.ForceDrawAllVelocities = InstancedView_ForceDrawAllVelocities;
	Result.DepthOfFieldSensorWidth = InstancedView_DepthOfFieldSensorWidth;
	Result.DepthOfFieldFocalDistance = InstancedView_DepthOfFieldFocalDistance;
	Result.DepthOfFieldScale = InstancedView_DepthOfFieldScale;
	Result.DepthOfFieldFocalLength = InstancedView_DepthOfFieldFocalLength;
	Result.DepthOfFieldFocalRegion = InstancedView_DepthOfFieldFocalRegion;
	Result.DepthOfFieldNearTransitionRegion = InstancedView_DepthOfFieldNearTransitionRegion;
	Result.DepthOfFieldFarTransitionRegion = InstancedView_DepthOfFieldFarTransitionRegion;
	Result.MotionBlurNormalizedToPixel = InstancedView_MotionBlurNormalizedToPixel;
	Result.GeneralPurposeTweak = InstancedView_GeneralPurposeTweak;
	Result.GeneralPurposeTweak2 = InstancedView_GeneralPurposeTweak2;
	Result.DemosaicVposOffset = InstancedView_DemosaicVposOffset;
	Result.DecalDepthBias = InstancedView_DecalDepthBias;
	Result.IndirectLightingColorScale = InstancedView_IndirectLightingColorScale;
	Result.PrecomputedIndirectLightingColorScale = InstancedView_PrecomputedIndirectLightingColorScale;
	Result.PrecomputedIndirectSpecularColorScale = InstancedView_PrecomputedIndirectSpecularColorScale;
	Result.AtmosphereLightDirection = InstancedView_AtmosphereLightDirection;
	Result.AtmosphereLightIlluminanceOnGroundPostTransmittance = InstancedView_AtmosphereLightIlluminanceOnGroundPostTransmittance;
	Result.AtmosphereLightIlluminanceOuterSpace = InstancedView_AtmosphereLightIlluminanceOuterSpace;
	Result.AtmosphereLightDiscLuminance = InstancedView_AtmosphereLightDiscLuminance;
	Result.AtmosphereLightDiscCosHalfApexAngle = InstancedView_AtmosphereLightDiscCosHalfApexAngle;
	Result.SkyViewLutSizeAndInvSize = InstancedView_SkyViewLutSizeAndInvSize;
	Result.SkyCameraTranslatedWorldOrigin = InstancedView_SkyCameraTranslatedWorldOrigin;
	Result.SkyPlanetTranslatedWorldCenterAndViewHeight = InstancedView_SkyPlanetTranslatedWorldCenterAndViewHeight;
	Result.SkyViewLutReferential = InstancedView_SkyViewLutReferential;
	Result.SkyAtmosphereSkyLuminanceFactor = InstancedView_SkyAtmosphereSkyLuminanceFactor;
	Result.SkyAtmospherePresentInScene = InstancedView_SkyAtmospherePresentInScene;
	Result.SkyAtmosphereHeightFogContribution = InstancedView_SkyAtmosphereHeightFogContribution;
	Result.SkyAtmosphereBottomRadiusKm = InstancedView_SkyAtmosphereBottomRadiusKm;
	Result.SkyAtmosphereTopRadiusKm = InstancedView_SkyAtmosphereTopRadiusKm;
	Result.SkyAtmosphereCameraAerialPerspectiveVolumeSizeAndInvSize = InstancedView_SkyAtmosphereCameraAerialPerspectiveVolumeSizeAndInvSize;
	Result.SkyAtmosphereAerialPerspectiveStartDepthKm = InstancedView_SkyAtmosphereAerialPerspectiveStartDepthKm;
	Result.SkyAtmosphereCameraAerialPerspectiveVolumeDepthResolution = InstancedView_SkyAtmosphereCameraAerialPerspectiveVolumeDepthResolution;
	Result.SkyAtmosphereCameraAerialPerspectiveVolumeDepthResolutionInv = InstancedView_SkyAtmosphereCameraAerialPerspectiveVolumeDepthResolutionInv;
	Result.SkyAtmosphereCameraAerialPerspectiveVolumeDepthSliceLengthKm = InstancedView_SkyAtmosphereCameraAerialPerspectiveVolumeDepthSliceLengthKm;
	Result.SkyAtmosphereCameraAerialPerspectiveVolumeDepthSliceLengthKmInv = InstancedView_SkyAtmosphereCameraAerialPerspectiveVolumeDepthSliceLengthKmInv;
	Result.SkyAtmosphereApplyCameraAerialPerspectiveVolume = InstancedView_SkyAtmosphereApplyCameraAerialPerspectiveVolume;
	Result.NormalCurvatureToRoughnessScaleBias = InstancedView_NormalCurvatureToRoughnessScaleBias;
	Result.RenderingReflectionCaptureMask = InstancedView_RenderingReflectionCaptureMask;
	Result.RealTimeReflectionCapture = InstancedView_RealTimeReflectionCapture;
	Result.RealTimeReflectionCapturePreExposure = InstancedView_RealTimeReflectionCapturePreExposure;
	Result.AmbientCubemapTint = InstancedView_AmbientCubemapTint;
	Result.AmbientCubemapIntensity = InstancedView_AmbientCubemapIntensity;
	Result.SkyLightApplyPrecomputedBentNormalShadowingFlag = InstancedView_SkyLightApplyPrecomputedBentNormalShadowingFlag;
	Result.SkyLightAffectReflectionFlag = InstancedView_SkyLightAffectReflectionFlag;
	Result.SkyLightAffectGlobalIlluminationFlag = InstancedView_SkyLightAffectGlobalIlluminationFlag;
	Result.SkyLightColor = InstancedView_SkyLightColor;
	Result.MobileSkyIrradianceEnvironmentMap = InstancedView_MobileSkyIrradianceEnvironmentMap;
	Result.MobilePreviewMode = InstancedView_MobilePreviewMode;
	Result.HMDEyePaddingOffset = InstancedView_HMDEyePaddingOffset;
	Result.ReflectionCubemapMaxMip = InstancedView_ReflectionCubemapMaxMip;
	Result.ShowDecalsMask = InstancedView_ShowDecalsMask;
	Result.DistanceFieldAOSpecularOcclusionMode = InstancedView_DistanceFieldAOSpecularOcclusionMode;
	Result.IndirectCapsuleSelfShadowingIntensity = InstancedView_IndirectCapsuleSelfShadowingIntensity;
	Result.ReflectionEnvironmentRoughnessMixingScaleBiasAndLargestWeight = InstancedView_ReflectionEnvironmentRoughnessMixingScaleBiasAndLargestWeight;
	Result.StereoPassIndex = InstancedView_StereoPassIndex;
	Result.GlobalVolumeCenterAndExtent = InstancedView_GlobalVolumeCenterAndExtent;
	Result.GlobalVolumeWorldToUVAddAndMul = InstancedView_GlobalVolumeWorldToUVAddAndMul;
	Result.GlobalDistanceFieldMipWorldToUVScale = InstancedView_GlobalDistanceFieldMipWorldToUVScale;
	Result.GlobalDistanceFieldMipWorldToUVBias = InstancedView_GlobalDistanceFieldMipWorldToUVBias;
	Result.GlobalDistanceFieldMipFactor = InstancedView_GlobalDistanceFieldMipFactor;
	Result.GlobalDistanceFieldMipTransition = InstancedView_GlobalDistanceFieldMipTransition;
	Result.GlobalDistanceFieldClipmapSizeInPages = InstancedView_GlobalDistanceFieldClipmapSizeInPages;
	Result.GlobalDistanceFieldInvPageAtlasSize = InstancedView_GlobalDistanceFieldInvPageAtlasSize;
	Result.GlobalDistanceFieldInvCoverageAtlasSize = InstancedView_GlobalDistanceFieldInvCoverageAtlasSize;
	Result.GlobalVolumeDimension = InstancedView_GlobalVolumeDimension;
	Result.GlobalVolumeTexelSize = InstancedView_GlobalVolumeTexelSize;
	Result.MaxGlobalDFAOConeDistance = InstancedView_MaxGlobalDFAOConeDistance;
	Result.NumGlobalSDFClipmaps = InstancedView_NumGlobalSDFClipmaps;
	Result.FullyCoveredExpandSurfaceScale = InstancedView_FullyCoveredExpandSurfaceScale;
	Result.UncoveredExpandSurfaceScale = InstancedView_UncoveredExpandSurfaceScale;
	Result.UncoveredMinStepScale = InstancedView_UncoveredMinStepScale;
	Result.CursorPosition = InstancedView_CursorPosition;
	Result.bCheckerboardSubsurfaceProfileRendering = InstancedView_bCheckerboardSubsurfaceProfileRendering;
	Result.VolumetricFogInvGridSize = InstancedView_VolumetricFogInvGridSize;
	Result.VolumetricFogGridZParams = InstancedView_VolumetricFogGridZParams;
	Result.VolumetricFogSVPosToVolumeUV = InstancedView_VolumetricFogSVPosToVolumeUV;
	Result.VolumetricFogMaxDistance = InstancedView_VolumetricFogMaxDistance;
	Result.VolumetricLightmapWorldToUVScale = InstancedView_VolumetricLightmapWorldToUVScale;
	Result.VolumetricLightmapWorldToUVAdd = InstancedView_VolumetricLightmapWorldToUVAdd;
	Result.VolumetricLightmapIndirectionTextureSize = InstancedView_VolumetricLightmapIndirectionTextureSize;
	Result.VolumetricLightmapBrickSize = InstancedView_VolumetricLightmapBrickSize;
	Result.VolumetricLightmapBrickTexelSize = InstancedView_VolumetricLightmapBrickTexelSize;
	Result.StereoIPD = InstancedView_StereoIPD;
	Result.IndirectLightingCacheShowFlag = InstancedView_IndirectLightingCacheShowFlag;
	Result.EyeToPixelSpreadAngle = InstancedView_EyeToPixelSpreadAngle;
	Result.XRPassthroughCameraUVs = InstancedView_XRPassthroughCameraUVs;
	Result.GlobalVirtualTextureMipBias = InstancedView_GlobalVirtualTextureMipBias;
	Result.VirtualTextureFeedbackShift = InstancedView_VirtualTextureFeedbackShift;
	Result.VirtualTextureFeedbackMask = InstancedView_VirtualTextureFeedbackMask;
	Result.VirtualTextureFeedbackStride = InstancedView_VirtualTextureFeedbackStride;
	Result.VirtualTextureFeedbackJitterOffset = InstancedView_VirtualTextureFeedbackJitterOffset;
	Result.VirtualTextureFeedbackSampleOffset = InstancedView_VirtualTextureFeedbackSampleOffset;
	Result.RuntimeVirtualTextureMipLevel = InstancedView_RuntimeVirtualTextureMipLevel;
	Result.RuntimeVirtualTexturePackHeight = InstancedView_RuntimeVirtualTexturePackHeight;
	Result.RuntimeVirtualTextureDebugParams = InstancedView_RuntimeVirtualTextureDebugParams;
	Result.OverrideLandscapeLOD = InstancedView_OverrideLandscapeLOD;
	Result.FarShadowStaticMeshLODBias = InstancedView_FarShadowStaticMeshLODBias;
	Result.MinRoughness = InstancedView_MinRoughness;
	Result.HairRenderInfo = InstancedView_HairRenderInfo;
	Result.EnableSkyLight = InstancedView_EnableSkyLight;
	Result.HairRenderInfoBits = InstancedView_HairRenderInfoBits;
	Result.HairComponents = InstancedView_HairComponents;
	Result.bSubsurfacePostprocessEnabled = InstancedView_bSubsurfacePostprocessEnabled;
	Result.SSProfilesTextureSizeAndInvSize = InstancedView_SSProfilesTextureSizeAndInvSize;
	Result.SSProfilesPreIntegratedTextureSizeAndInvSize = InstancedView_SSProfilesPreIntegratedTextureSizeAndInvSize;
	Result.PhysicsFieldClipmapCenter = InstancedView_PhysicsFieldClipmapCenter;
	Result.PhysicsFieldClipmapDistance = InstancedView_PhysicsFieldClipmapDistance;
	Result.PhysicsFieldClipmapResolution = InstancedView_PhysicsFieldClipmapResolution;
	Result.PhysicsFieldClipmapExponent = InstancedView_PhysicsFieldClipmapExponent;
	Result.PhysicsFieldClipmapCount = InstancedView_PhysicsFieldClipmapCount;
	Result.PhysicsFieldTargetCount = InstancedView_PhysicsFieldTargetCount;
	Result.PhysicsFieldTargets = InstancedView_PhysicsFieldTargets;
	Result.InstanceSceneDataSOAStride = InstancedView_InstanceSceneDataSOAStride;
	Result.GPUSceneViewId = InstancedView_GPUSceneViewId;
	FinalizeViewState(Result);
	return Result;
}
#line 16 "/Engine/Private/InstancedStereo.ush"

void FinalizeViewState(inout ViewState InOutView)
{
	InOutView.WorldToClip = MakeLWCInverseMatrix(InOutView.MatrixTilePosition, InOutView.RelativeWorldToClip);
	InOutView.ClipToWorld = MakeLWCMatrix(InOutView.MatrixTilePosition, InOutView.ClipToRelativeWorld);
	InOutView.ScreenToWorld = MakeLWCMatrix(InOutView.MatrixTilePosition, InOutView.ScreenToRelativeWorld);
	InOutView.PrevClipToWorld = MakeLWCMatrix(InOutView.MatrixTilePosition, InOutView.PrevClipToRelativeWorld);

	InOutView.WorldCameraOrigin = MakeLWCVector3(InOutView.ViewTilePosition, InOutView.RelativeWorldCameraOrigin);
	InOutView.WorldViewOrigin = MakeLWCVector3(InOutView.ViewTilePosition, InOutView.RelativeWorldViewOrigin);
	InOutView.PrevWorldCameraOrigin = MakeLWCVector3(InOutView.ViewTilePosition, InOutView.PrevRelativeWorldCameraOrigin);
	InOutView.PrevWorldViewOrigin = MakeLWCVector3(InOutView.ViewTilePosition, InOutView.PrevRelativeWorldViewOrigin);
	InOutView.PreViewTranslation = MakeLWCVector3(-InOutView.ViewTilePosition, InOutView.RelativePreViewTranslation);
	InOutView.PrevPreViewTranslation = MakeLWCVector3(-InOutView.ViewTilePosition, InOutView.RelativePrevPreViewTranslation);
}



static ViewState ResolvedView = (ViewState)0.0f;

ViewState ResolveView()
{
	return GetPrimaryView();
}
#line 61 "/Engine/Private/InstancedStereo.ush"
bool IsInstancedStereo()
{



	return false;

}

uint GetEyeIndex(uint InstanceId)
{



	return 0;

}

uint GetInstanceId(uint InstanceId)
{



	return InstanceId;

}
#line 199 "/Engine/Private/Common.ush"
#line 200 "/Engine/Private/Common.ush"
#line 1 "Definitions.usf"
#line 201 "/Engine/Private/Common.ush"
#line 202 "/Engine/Private/Common.ush"
#line 1 "AssertionMacros.ush"
#line 203 "/Engine/Private/Common.ush"
#line 235 "/Engine/Private/Common.ush"
static float GlobalTextureMipBias = 0;
static float GlobalRayCone_TexArea = 0;
float ComputeRayConeLod(Texture2D Tex)
{






    return  0.0f ;

}

float ClampToHalfFloatRange(float X) { return clamp(X, float(0), MaxHalfFloat); }
float2 ClampToHalfFloatRange(float2 X) { return clamp(X, float(0).xx, MaxHalfFloat.xx); }
float3 ClampToHalfFloatRange(float3 X) { return clamp(X, float(0).xxx, MaxHalfFloat.xxx); }
float4 ClampToHalfFloatRange(float4 X) { return clamp(X, float(0).xxxx, MaxHalfFloat.xxxx); }



float4  Texture1DSample(Texture1D Tex, SamplerState Sampler, float UV)
{

	return Tex.SampleLevel(Sampler, UV, 0);
#line 263 "/Engine/Private/Common.ush"
}
float4  Texture2DSample(Texture2D Tex, SamplerState Sampler, float2 UV)
{

	return Tex.SampleLevel(Sampler, UV, ComputeRayConeLod(Tex) + GlobalTextureMipBias);
#line 271 "/Engine/Private/Common.ush"
}
float4  Texture2DSample(Texture2D Tex, SamplerState Sampler, FloatDeriv2 UV)
{

	return Tex.SampleLevel(Sampler, UV.Value, ComputeRayConeLod(Tex) + GlobalTextureMipBias);
#line 279 "/Engine/Private/Common.ush"
}
float  Texture2DSample_A8(Texture2D Tex, SamplerState Sampler, float2 UV)
{

	return Tex.SampleLevel(Sampler, UV, ComputeRayConeLod(Tex) + GlobalTextureMipBias)  .a ;
#line 287 "/Engine/Private/Common.ush"
}
float4  Texture3DSample(Texture3D Tex, SamplerState Sampler, float3 UV)
{

	return Tex.SampleLevel(Sampler, UV, 0);
#line 295 "/Engine/Private/Common.ush"
}
float4  TextureCubeSample(TextureCube Tex, SamplerState Sampler, float3 UV)
{

	return Tex.SampleLevel(Sampler, UV, 0);
#line 303 "/Engine/Private/Common.ush"
}
float4  Texture2DArraySample(Texture2DArray Tex, SamplerState Sampler, float3 UV)
{

	return Tex.SampleLevel(Sampler, UV, 0);
#line 311 "/Engine/Private/Common.ush"
}
float4  Texture1DSampleLevel(Texture1D Tex, SamplerState Sampler, float UV,  float  Mip)
{
	return Tex.SampleLevel(Sampler, UV, Mip);
}
float4  Texture2DSampleLevel(Texture2D Tex, SamplerState Sampler, float2 UV,  float  Mip)
{
	return Tex.SampleLevel(Sampler, UV, Mip);
}
float4  Texture2DSampleBias(Texture2D Tex, SamplerState Sampler, float2 UV,  float  MipBias)
{

	return Tex.SampleLevel(Sampler, UV, ComputeRayConeLod(Tex) + MipBias + GlobalTextureMipBias);
#line 327 "/Engine/Private/Common.ush"
}
float4  Texture2DSampleGrad(Texture2D Tex, SamplerState Sampler, float2 UV,  float2  DDX,  float2  DDY)
{
	return Tex.SampleGrad(Sampler, UV, DDX, DDY);
}
float4  Texture3DSampleLevel(Texture3D Tex, SamplerState Sampler, float3 UV,  float  Mip)
{
	return Tex.SampleLevel(Sampler, UV, Mip);
}
float4  Texture3DSampleBias(Texture3D Tex, SamplerState Sampler, float3 UV,  float  MipBias)
{

	return Tex.SampleLevel(Sampler, UV, 0);
#line 343 "/Engine/Private/Common.ush"
}
float4  Texture3DSampleGrad(Texture3D Tex, SamplerState Sampler, float3 UV,  float3  DDX,  float3  DDY)
{
	return Tex.SampleGrad(Sampler, UV, DDX, DDY);
}
float4  TextureCubeSampleLevel(TextureCube Tex, SamplerState Sampler, float3 UV,  float  Mip)
{
	return Tex.SampleLevel(Sampler, UV, Mip);
}
float  TextureCubeSampleDepthLevel(TextureCube TexDepth, SamplerState Sampler, float3 UV,  float  Mip)
{
	return TexDepth.SampleLevel(Sampler, UV, Mip).x;
}
float4  TextureCubeSampleBias(TextureCube Tex, SamplerState Sampler, float3 UV,  float  MipBias)
{

	return Tex.SampleLevel(Sampler, UV, 0);
#line 363 "/Engine/Private/Common.ush"
}
float4  TextureCubeSampleGrad(TextureCube Tex, SamplerState Sampler, float3 UV,  float3  DDX,  float3  DDY)
{
	return Tex.SampleGrad(Sampler, UV, DDX, DDY);
}
float4  TextureExternalSample( Texture2D  Tex, SamplerState Sampler, float2 UV)
{




		return Tex.SampleLevel(Sampler, UV, ComputeRayConeLod(Tex) + GlobalTextureMipBias);
#line 379 "/Engine/Private/Common.ush"
}
float4  TextureExternalSampleGrad( Texture2D  Tex, SamplerState Sampler, float2 UV,  float2  DDX,  float2  DDY)
{
	return Tex.SampleGrad(Sampler, UV, DDX, DDY);
}
float4  TextureExternalSampleLevel( Texture2D  Tex, SamplerState Sampler, float2 UV,  float  Mip)
{
	return Tex.SampleLevel(Sampler, UV, Mip);
}




float4  Texture1DSample_Decal(Texture1D Tex, SamplerState Sampler, float UV)
{



	return Texture1DSample(Tex, Sampler, UV);

}
float4  Texture2DSample_Decal(Texture2D Tex, SamplerState Sampler, float2 UV)
{



	return Texture2DSample(Tex, Sampler, UV);

}
float4  Texture3DSample_Decal(Texture3D Tex, SamplerState Sampler, float3 UV)
{



	return Texture3DSample(Tex, Sampler, UV);

}
float4  TextureCubeSample_Decal(TextureCube Tex, SamplerState Sampler, float3 UV)
{



	return TextureCubeSample(Tex, Sampler, UV);

}
float4  TextureExternalSample_Decal( Texture2D  Tex, SamplerState Sampler, float2 UV)
{



	return TextureExternalSample(Tex, Sampler, UV);

}

float4  Texture2DArraySampleLevel(Texture2DArray Tex, SamplerState Sampler, float3 UV,  float  Mip)
{
	return Tex.SampleLevel(Sampler, UV, Mip);
}
float4  Texture2DArraySampleBias(Texture2DArray Tex, SamplerState Sampler, float3 UV,  float  MipBias)
{

	return Tex.SampleLevel(Sampler, UV, 0);
#line 444 "/Engine/Private/Common.ush"
}
float4  Texture2DArraySampleGrad(Texture2DArray Tex, SamplerState Sampler, float3 UV,  float2  DDX,  float2  DDY)
{
	return Tex.SampleGrad(Sampler, UV, DDX, DDY);
}


float2 Tile1Dto2D(float xsize, float idx)
{
	float2 xyidx = 0;
	xyidx.y = floor(idx / xsize);
	xyidx.x = idx - xsize * xyidx.y;

	return xyidx;
}
#line 471 "/Engine/Private/Common.ush"
float4 PseudoVolumeTexture(Texture2D Tex, SamplerState TexSampler, float3 inPos, float2 xysize, float numframes,
	uint mipmode = 0, float miplevel = 0, float2 InDDX = 0, float2 InDDY = 0)
{
	float z = inPos.z - 0.5f / numframes;
	float zframe = floor(z * numframes);
	float zphase = frac(z * numframes);

	float2 uv = frac(inPos.xy) / xysize;

	float2 curframe = Tile1Dto2D(xysize.x, zframe) / xysize;
	float2 nextframe = Tile1Dto2D(xysize.x, zframe + 1) / xysize;

	float2 uvCurFrame = uv + curframe;
	float2 uvNextFrame = uv + nextframe;
#line 491 "/Engine/Private/Common.ush"
	float4 sampleA = 0, sampleB = 0;
	switch (mipmode)
	{
	case 0:
		sampleA = Tex.SampleLevel(TexSampler, uvCurFrame, miplevel);
		sampleB = Tex.SampleLevel(TexSampler, uvNextFrame, miplevel);
		break;
	case 1:
		sampleA = Texture2DSample(Tex, TexSampler, uvCurFrame);
		sampleB = Texture2DSample(Tex, TexSampler, uvNextFrame);
		break;
	case 2:
		sampleA = Tex.SampleGrad(TexSampler, uvCurFrame, InDDX, InDDY);
		sampleB = Tex.SampleGrad(TexSampler, uvNextFrame, InDDX, InDDY);
		break;
	default:
		break;
	}

	return lerp(sampleA, sampleB, zphase);
}


float4  TextureCubeArraySample(TextureCubeArray Tex, SamplerState Sampler, float4 UV)
{
	return Tex.Sample(Sampler, UV);
}

float4  TextureCubeArraySampleLevel(TextureCubeArray Tex, SamplerState Sampler, float4 UV,  float  Mip)
{
	return Tex.SampleLevel(Sampler, UV, Mip);
}

float4  TextureCubeArraySampleBias(TextureCubeArray Tex, SamplerState Sampler, float4 UV,  float  MipBias)
{

	return Tex.SampleLevel(Sampler, UV, 0);
#line 531 "/Engine/Private/Common.ush"
}

float4  TextureCubeArraySampleGrad(TextureCubeArray Tex, SamplerState Sampler, float4 UV,  float3  DDX,  float3  DDY)
{
	return Tex.SampleGrad(Sampler, UV, DDX, DDY);
}


float4  TextureCubeArraySampleLevel(TextureCubeArray Tex, SamplerState Sampler, float3 UV, float ArrayIndex,  float  Mip)
{
	return TextureCubeArraySampleLevel(Tex, Sampler, float4(UV, ArrayIndex), Mip);
}
#line 582 "/Engine/Private/Common.ush"
float  Luminance(  float3  LinearColor )
{
	return dot( LinearColor,  float3 ( 0.3, 0.59, 0.11 ) );
}

float  length2( float2  v)
{
	return dot(v, v);
}
float  length2( float3  v)
{
	return dot(v, v);
}
float  length2( float4  v)
{
	return dot(v, v);
}

uint Mod(uint a, uint b)
{

	return a % b;
#line 607 "/Engine/Private/Common.ush"
}

uint2 Mod(uint2 a, uint2 b)
{

	return a % b;
#line 616 "/Engine/Private/Common.ush"
}

uint3 Mod(uint3 a, uint3 b)
{

	return a % b;
#line 625 "/Engine/Private/Common.ush"
}

float  UnClampedPow( float  X,  float  Y)
{
	return pow(X,  Y );
}
float2  UnClampedPow( float2  X,  float2  Y)
{
	return pow(X,  Y );
}
float3  UnClampedPow( float3  X,  float3  Y)
{
	return pow(X,  Y );
}
float4  UnClampedPow( float4  X,  float4  Y)
{
	return pow(X,  Y );
}




float  ClampedPow( float  X, float  Y)
{
	return pow(max(abs(X), 0.000001f ),Y);
}
float2  ClampedPow( float2  X, float2  Y)
{
	return pow(max(abs(X), float2 ( 0.000001f , 0.000001f )),Y);
}
float3  ClampedPow( float3  X, float3  Y)
{
	return pow(max(abs(X), float3 ( 0.000001f , 0.000001f , 0.000001f )),Y);
}
float4  ClampedPow( float4  X, float4  Y)
{
	return pow(max(abs(X), float4 ( 0.000001f , 0.000001f , 0.000001f , 0.000001f )),Y);
}


float  PositiveClampedPow( float  Base,  float  Exponent)
{
	return (Base <= 0.0f) ? 0.0f : pow(Base, Exponent);
}
float2  PositiveClampedPow( float2  Base,  float2  Exponent)
{
	return  float2 (PositiveClampedPow(Base.x, Exponent.x), PositiveClampedPow(Base.y, Exponent.y));
}
float3  PositiveClampedPow( float3  Base,  float3  Exponent)
{
	return  float3 (PositiveClampedPow(Base.xy, Exponent.xy), PositiveClampedPow(Base.z, Exponent.z));
}
float4  PositiveClampedPow( float4  Base,  float4  Exponent)
{
	return  float4 (PositiveClampedPow(Base.xy, Exponent.xy), PositiveClampedPow(Base.zw, Exponent.zw));
}

float DDX(float Input)
{

	return 0;
#line 689 "/Engine/Private/Common.ush"
}

float2 DDX(float2 Input)
{

	return 0;
#line 698 "/Engine/Private/Common.ush"
}

float3 DDX(float3 Input)
{

	return 0;
#line 707 "/Engine/Private/Common.ush"
}

float4 DDX(float4 Input)
{

	return 0;
#line 716 "/Engine/Private/Common.ush"
}

float DDY(float Input)
{

	return 0;
#line 725 "/Engine/Private/Common.ush"
}

float2 DDY(float2 Input)
{

	return 0;
#line 734 "/Engine/Private/Common.ush"
}

float3 DDY(float3 Input)
{

	return 0;
#line 743 "/Engine/Private/Common.ush"
}

float4 DDY(float4 Input)
{

	return 0;
#line 752 "/Engine/Private/Common.ush"
}
#line 754 "/Engine/Private/Common.ush"
#line 1 "FastMath.ush"
#line 46 "/Engine/Private/FastMath.ush"
float rsqrtFast( float x )
{
	int i = asint(x);
	i = 0x5f3759df - (i >> 1);
	return asfloat(i);
}




float sqrtFast( float x )
{
	int i = asint(x);
	i = 0x1FBD1DF5 + (i >> 1);
	return asfloat(i);
}




float rcpFast( float x )
{
	int i = asint(x);
	i = 0x7EF311C2 - i;
	return asfloat(i);
}





float rcpFastNR1( float x )
{
	int i = asint(x);
	i = 0x7EF311C3 - i;
	float xRcp = asfloat(i);
	xRcp = xRcp * (-xRcp * x + 2.0f);
	return xRcp;
}

float lengthFast( float3 v )
{
	float LengthSqr = dot(v,v);
	return sqrtFast( LengthSqr );
}

float3 normalizeFast( float3 v )
{
	float LengthSqr = dot(v,v);
	return v * rsqrtFast( LengthSqr );
}

float4 fastClamp(float4 x, float4 Min, float4 Max)
{




	return clamp(x, Min, Max);

}

float3 fastClamp(float3 x, float3 Min, float3 Max)
{




	return clamp(x, Min, Max);

}

float2 fastClamp(float2 x, float2 Min, float2 Max)
{




	return clamp(x, Min, Max);

}

float fastClamp(float x, float Min, float Max)
{




	return clamp(x, Min, Max);

}

int4 fastClamp(int4 x, int4 Min, int4 Max)
{




	return clamp(x, Min, Max);

}

int3 fastClamp(int3 x, int3 Min, int3 Max)
{




	return clamp(x, Min, Max);

}

int2 fastClamp(int2 x, int2 Min, int2 Max)
{




	return clamp(x, Min, Max);

}

int fastClamp(int x, int Min, int Max)
{




	return clamp(x, Min, Max);

}









float acosFast(float inX)
{
    float x = abs(inX);
    float res = -0.156583f * x + (0.5 * PI);
    res *= sqrt(1.0f - x);
    return (inX >= 0) ? res : PI - res;
}

float2 acosFast( float2 x )
{
	return float2( acosFast(x.x), acosFast(x.y) );
}

float3 acosFast( float3 x )
{
	return float3( acosFast(x.x), acosFast(x.y), acosFast(x.z) );
}

float4 acosFast( float4 x )
{
	return float4( acosFast(x.x), acosFast(x.y), acosFast(x.z), acosFast(x.w) );
}




float asinFast( float x )
{
    return (0.5 * PI) - acosFast(x);
}

float2 asinFast( float2 x)
{
	return float2( asinFast(x.x), asinFast(x.y) );
}

float3 asinFast( float3 x)
{
	return float3( asinFast(x.x), asinFast(x.y), asinFast(x.z) );
}

float4 asinFast( float4 x )
{
	return float4( asinFast(x.x), asinFast(x.y), asinFast(x.z), asinFast(x.w) );
}





float atanFastPos( float x )
{
    float t0 = (x < 1.0f) ? x : 1.0f / x;
    float t1 = t0 * t0;
    float poly = 0.0872929f;
    poly = -0.301895f + poly * t1;
    poly = 1.0f + poly * t1;
    poly = poly * t0;
    return (x < 1.0f) ? poly : (0.5 * PI) - poly;
}



float atanFast( float x )
{
    float t0 = atanFastPos( abs(x) );
    return (x < 0) ? -t0: t0;
}

float2 atanFast( float2 x )
{
	return float2( atanFast(x.x), atanFast(x.y) );
}

float3 atanFast( float3 x )
{
	return float3( atanFast(x.x), atanFast(x.y), atanFast(x.z) );
}

float4 atanFast( float4 x )
{
	return float4( atanFast(x.x), atanFast(x.y), atanFast(x.z), atanFast(x.w) );
}

float atan2Fast( float y, float x )
{
	float t0 = max( abs(x), abs(y) );
	float t1 = min( abs(x), abs(y) );
	float t3 = t1 / t0;
	float t4 = t3 * t3;


	t0 = + 0.0872929;
	t0 = t0 * t4 - 0.301895;
	t0 = t0 * t4 + 1.0;
	t3 = t0 * t3;

	t3 = abs(y) > abs(x) ? (0.5 * PI) - t3 : t3;
	t3 = x < 0 ? PI - t3 : t3;
	t3 = y < 0 ? -t3 : t3;

	return t3;
}

float2 atan2Fast( float2 y, float2 x )
{
	return float2( atan2Fast(y.x, x.x), atan2Fast(y.y, x.y) );
}

float3 atan2Fast( float3 y, float3 x )
{
	return float3( atan2Fast(y.x, x.x), atan2Fast(y.y, x.y), atan2Fast(y.z, x.z) );
}

float4 atan2Fast( float4 y, float4 x )
{
	return float4( atan2Fast(y.x, x.x), atan2Fast(y.y, x.y), atan2Fast(y.z, x.z), atan2Fast(y.w, x.w) );
}





float acosFast4(float inX)
{
	float x1 = abs(inX);
	float x2 = x1 * x1;
	float x3 = x2 * x1;
	float s;

	s = -0.2121144f * x1 + 1.5707288f;
	s = 0.0742610f * x2 + s;
	s = -0.0187293f * x3 + s;
	s = sqrt(1.0f - x1) * s;



	return inX >= 0.0f ? s : PI - s;
}




float asinFast4( float x )
{
	return (0.5 * PI) - acosFast4(x);
}




float CosBetweenVectors(float3 A, float3 B)
{

	return dot(A, B) * rsqrt(length2(A) * length2(B));
}



float AngleBetweenVectors(float3 A, float3 B)
{
	return acos(CosBetweenVectors(A, B));
}


float AngleBetweenVectorsFast(float3 A, float3 B)
{
	return acosFast(CosBetweenVectors(A, B));
}


int SignFastInt(float v)
{
	return 1 - int((asuint(v) & 0x80000000) >> 30);
}

int2 SignFastInt(float2 v)
{
	return int2(SignFastInt(v.x), SignFastInt(v.y));
}
#line 755 "/Engine/Private/Common.ush"
#line 1 "Random.ush"
#line 12 "/Engine/Private/Random.ush"
float PseudoRandom(float2 xy)
{
	float2 pos = frac(xy / 128.0f) * 128.0f + float2(-64.340622f, -72.465622f);


	return frac(dot(pos.xyx * pos.xyy, float3(20.390625f, 60.703125f, 2.4281209f)));
}







float InterleavedGradientNoise( float2 uv, float FrameId )
{

	uv += FrameId * (float2(47, 17) * 0.695f);

    const float3 magic = float3( 0.06711056f, 0.00583715f, 52.9829189f );
    return frac(magic.z * frac(dot(uv, magic.xy)));
}



float RandFast( uint2 PixelPos, float Magic = 3571.0 )
{
	float2 Random2 = ( 1.0 / 4320.0 ) * PixelPos + float2( 0.25, 0.0 );
	float Random = frac( dot( Random2 * Random2, Magic ) );
	Random = frac( Random * Random * (2 * Magic) );
	return Random;
}
#line 56 "/Engine/Private/Random.ush"
float RandBBSfloat(float seed)
{
	float s = frac(seed /  4093 );
	s = frac(s * s *  4093 );
	s = frac(s * s *  4093 );
	return s;
}








uint3 Rand3DPCG16(int3 p)
{

	uint3 v = uint3(p);




	v = v * 1664525u + 1013904223u;
#line 94 "/Engine/Private/Random.ush"
	v.x += v.y*v.z;
	v.y += v.z*v.x;
	v.z += v.x*v.y;
	v.x += v.y*v.z;
	v.y += v.z*v.x;
	v.z += v.x*v.y;


	return v >> 16u;
}






uint3 Rand3DPCG32(int3 p)
{

	uint3 v = uint3(p);


	v = v * 1664525u + 1013904223u;


	v.x += v.y*v.z;
	v.y += v.z*v.x;
	v.z += v.x*v.y;


	v ^= v >> 16u;


	v.x += v.y*v.z;
	v.y += v.z*v.x;
	v.z += v.x*v.y;

	return v;
}








uint4 Rand4DPCG32(int4 p)
{

	uint4 v = uint4(p);


	v = v * 1664525u + 1013904223u;


	v.x += v.y*v.w;
	v.y += v.z*v.x;
	v.z += v.x*v.y;
	v.w += v.y*v.z;


	v ^= (v >> 16u);


	v.x += v.y*v.w;
	v.y += v.z*v.x;
	v.z += v.x*v.y;
	v.w += v.y*v.z;

	return v;
}
#line 174 "/Engine/Private/Random.ush"
void FindBestAxisVectors(float3 In, out float3 Axis1, out float3 Axis2 )
{
	const float3 N = abs(In);


	if( N.z > N.x && N.z > N.y )
	{
		Axis1 = float3(1, 0, 0);
	}
	else
	{
		Axis1 = float3(0, 0, 1);
	}

	Axis1 = normalize(Axis1 - In * dot(Axis1, In));
	Axis2 = cross(Axis1, In);
}
#line 215 "/Engine/Private/Random.ush"
uint2 ScrambleTEA(uint2 v, uint IterationCount = 3)
{

	uint k[4] ={ 0xA341316Cu , 0xC8013EA4u , 0xAD90777Du , 0x7E95761Eu };

	uint y = v[0];
	uint z = v[1];
	uint sum = 0;

	[unroll]  for(uint i = 0; i < IterationCount; ++i)
	{
		sum += 0x9e3779b9;
		y += ((z << 4u) + k[0]) ^ (z + sum) ^ ((z >> 5u) + k[1]);
		z += ((y << 4u) + k[2]) ^ (y + sum) ^ ((y >> 5u) + k[3]);
	}

	return uint2(y, z);
}






float3 NoiseTileWrap(float3 v, bool bTiling, float RepeatSize)
{
	return bTiling ? (frac(v / RepeatSize) * RepeatSize) : v;
}




float4 PerlinRamp(float4 t)
{
	return t * t * t * (t * (t * 6 - 15) + 10);
}




float4 PerlinRampDerivative(float4 t)
{
	return t * t * (t * (t * 30 - 60) + 30);
}







float4 MGradient(int seed, float3 offset)
{
	uint rand = Rand3DPCG16(int3(seed,0,0)).x;
	float3 direction = float3(rand.xxx &  int3(0x8000, 0x4000, 0x2000) ) *  float3(1. / 0x4000, 1. / 0x2000, 1. / 0x1000)  - 1;
	return float4(direction, dot(direction, offset));
}







float3 NoiseSeeds(float3 v, bool bTiling, float RepeatSize,
	out float seed000, out float seed001, out float seed010, out float seed011,
	out float seed100, out float seed101, out float seed110, out float seed111)
{
	float3 fv = frac(v);
	float3 iv = floor(v);

	const float3 primes = float3(19, 47, 101);

	if (bTiling)
	{
		seed000 = dot(primes, NoiseTileWrap(iv, true, RepeatSize));
		seed100 = dot(primes, NoiseTileWrap(iv + float3(1, 0, 0), true, RepeatSize));
		seed010 = dot(primes, NoiseTileWrap(iv + float3(0, 1, 0), true, RepeatSize));
		seed110 = dot(primes, NoiseTileWrap(iv + float3(1, 1, 0), true, RepeatSize));
		seed001 = dot(primes, NoiseTileWrap(iv + float3(0, 0, 1), true, RepeatSize));
		seed101 = dot(primes, NoiseTileWrap(iv + float3(1, 0, 1), true, RepeatSize));
		seed011 = dot(primes, NoiseTileWrap(iv + float3(0, 1, 1), true, RepeatSize));
		seed111 = dot(primes, NoiseTileWrap(iv + float3(1, 1, 1), true, RepeatSize));
	}
	else
	{
		seed000 = dot(iv, primes);
		seed100 = seed000 + primes.x;
		seed010 = seed000 + primes.y;
		seed110 = seed100 + primes.y;
		seed001 = seed000 + primes.z;
		seed101 = seed100 + primes.z;
		seed011 = seed010 + primes.z;
		seed111 = seed110 + primes.z;
	}

	return fv;
}







float GradientNoise3D_ALU(float3 v, bool bTiling, float RepeatSize)
{
	float seed000, seed001, seed010, seed011, seed100, seed101, seed110, seed111;
	float3 fv = NoiseSeeds(v, bTiling, RepeatSize, seed000, seed001, seed010, seed011, seed100, seed101, seed110, seed111);

	float rand000 = MGradient(int(seed000), fv - float3(0, 0, 0)).w;
	float rand100 = MGradient(int(seed100), fv - float3(1, 0, 0)).w;
	float rand010 = MGradient(int(seed010), fv - float3(0, 1, 0)).w;
	float rand110 = MGradient(int(seed110), fv - float3(1, 1, 0)).w;
	float rand001 = MGradient(int(seed001), fv - float3(0, 0, 1)).w;
	float rand101 = MGradient(int(seed101), fv - float3(1, 0, 1)).w;
	float rand011 = MGradient(int(seed011), fv - float3(0, 1, 1)).w;
	float rand111 = MGradient(int(seed111), fv - float3(1, 1, 1)).w;

	float3 Weights = PerlinRamp(float4(fv, 0)).xyz;

	float i = lerp(lerp(rand000, rand100, Weights.x), lerp(rand010, rand110, Weights.x), Weights.y);
	float j = lerp(lerp(rand001, rand101, Weights.x), lerp(rand011, rand111, Weights.x), Weights.y);
	return lerp(i, j, Weights.z).x;
}





float4x3 SimplexCorners(float3 v)
{

	float3 tet = floor(v + v.x/3 + v.y/3 + v.z/3);
	float3 base = tet - tet.x/6 - tet.y/6 - tet.z/6;
	float3 f = v - base;



	float3 g = step(f.yzx, f.xyz), h = 1 - g.zxy;
	float3 a1 = min(g, h) - 1. / 6., a2 = max(g, h) - 1. / 3.;


	return float4x3(base, base + a1, base + a2, base + 0.5);
}




float4 SimplexSmooth(float4x3 f)
{
	const float scale = 1024. / 375.;
	float4 d = float4(dot(f[0], f[0]), dot(f[1], f[1]), dot(f[2], f[2]), dot(f[3], f[3]));
	float4 s = saturate(2 * d);
	return (1 * scale + s*(-3 * scale + s*(3 * scale - s*scale)));
}




float3x4 SimplexDSmooth(float4x3 f)
{
	const float scale = 1024. / 375.;
	float4 d = float4(dot(f[0], f[0]), dot(f[1], f[1]), dot(f[2], f[2]), dot(f[3], f[3]));
	float4 s = saturate(2 * d);
	s = -12 * scale + s*(24 * scale - s * 12 * scale);

	return float3x4(
		s * float4(f[0][0], f[1][0], f[2][0], f[3][0]),
		s * float4(f[0][1], f[1][1], f[2][1], f[3][1]),
		s * float4(f[0][2], f[1][2], f[2][2], f[3][2]));
}
#line 403 "/Engine/Private/Random.ush"
float3x4 JacobianSimplex_ALU(float3 v, bool bTiling, float RepeatSize)
{

	float4x3 T = SimplexCorners(v);
	uint3 rand;
	float4x3 gvec[3], fv;
	float3x4 grad;



	fv[0] = v - T[0];
	rand = Rand3DPCG16(int3(floor(NoiseTileWrap(6 * T[0] + 0.5, bTiling, RepeatSize))));
	gvec[0][0] = float3(rand.xxx &  int3(0x8000, 0x4000, 0x2000) ) *  float3(1. / 0x4000, 1. / 0x2000, 1. / 0x1000)  - 1;
	gvec[1][0] = float3(rand.yyy &  int3(0x8000, 0x4000, 0x2000) ) *  float3(1. / 0x4000, 1. / 0x2000, 1. / 0x1000)  - 1;
	gvec[2][0] = float3(rand.zzz &  int3(0x8000, 0x4000, 0x2000) ) *  float3(1. / 0x4000, 1. / 0x2000, 1. / 0x1000)  - 1;
	grad[0][0] = dot(gvec[0][0], fv[0]);
	grad[1][0] = dot(gvec[1][0], fv[0]);
	grad[2][0] = dot(gvec[2][0], fv[0]);

	fv[1] = v - T[1];
	rand = Rand3DPCG16(int3(floor(NoiseTileWrap(6 * T[1] + 0.5, bTiling, RepeatSize))));
	gvec[0][1] = float3(rand.xxx &  int3(0x8000, 0x4000, 0x2000) ) *  float3(1. / 0x4000, 1. / 0x2000, 1. / 0x1000)  - 1;
	gvec[1][1] = float3(rand.yyy &  int3(0x8000, 0x4000, 0x2000) ) *  float3(1. / 0x4000, 1. / 0x2000, 1. / 0x1000)  - 1;
	gvec[2][1] = float3(rand.zzz &  int3(0x8000, 0x4000, 0x2000) ) *  float3(1. / 0x4000, 1. / 0x2000, 1. / 0x1000)  - 1;
	grad[0][1] = dot(gvec[0][1], fv[1]);
	grad[1][1] = dot(gvec[1][1], fv[1]);
	grad[2][1] = dot(gvec[2][1], fv[1]);

	fv[2] = v - T[2];
	rand = Rand3DPCG16(int3(floor(NoiseTileWrap(6 * T[2] + 0.5, bTiling, RepeatSize))));
	gvec[0][2] = float3(rand.xxx &  int3(0x8000, 0x4000, 0x2000) ) *  float3(1. / 0x4000, 1. / 0x2000, 1. / 0x1000)  - 1;
	gvec[1][2] = float3(rand.yyy &  int3(0x8000, 0x4000, 0x2000) ) *  float3(1. / 0x4000, 1. / 0x2000, 1. / 0x1000)  - 1;
	gvec[2][2] = float3(rand.zzz &  int3(0x8000, 0x4000, 0x2000) ) *  float3(1. / 0x4000, 1. / 0x2000, 1. / 0x1000)  - 1;
	grad[0][2] = dot(gvec[0][2], fv[2]);
	grad[1][2] = dot(gvec[1][2], fv[2]);
	grad[2][2] = dot(gvec[2][2], fv[2]);

	fv[3] = v - T[3];
	rand = Rand3DPCG16(int3(floor(NoiseTileWrap(6 * T[3] + 0.5, bTiling, RepeatSize))));
	gvec[0][3] = float3(rand.xxx &  int3(0x8000, 0x4000, 0x2000) ) *  float3(1. / 0x4000, 1. / 0x2000, 1. / 0x1000)  - 1;
	gvec[1][3] = float3(rand.yyy &  int3(0x8000, 0x4000, 0x2000) ) *  float3(1. / 0x4000, 1. / 0x2000, 1. / 0x1000)  - 1;
	gvec[2][3] = float3(rand.zzz &  int3(0x8000, 0x4000, 0x2000) ) *  float3(1. / 0x4000, 1. / 0x2000, 1. / 0x1000)  - 1;
	grad[0][3] = dot(gvec[0][3], fv[3]);
	grad[1][3] = dot(gvec[1][3], fv[3]);
	grad[2][3] = dot(gvec[2][3], fv[3]);


	float4 sv = SimplexSmooth(fv);
	float3x4 ds = SimplexDSmooth(fv);

	float3x4 jacobian;
	jacobian[0] = float4(mul(sv, gvec[0]) + mul(ds, grad[0]), dot(sv, grad[0]));
	jacobian[1] = float4(mul(sv, gvec[1]) + mul(ds, grad[1]), dot(sv, grad[1]));
	jacobian[2] = float4(mul(sv, gvec[2]) + mul(ds, grad[2]), dot(sv, grad[2]));

	return jacobian;
}






float ValueNoise3D_ALU(float3 v, bool bTiling, float RepeatSize)
{
	float seed000, seed001, seed010, seed011, seed100, seed101, seed110, seed111;
	float3 fv = NoiseSeeds(v, bTiling, RepeatSize, seed000, seed001, seed010, seed011, seed100, seed101, seed110, seed111);

	float rand000 = RandBBSfloat(seed000) * 2 - 1;
	float rand100 = RandBBSfloat(seed100) * 2 - 1;
	float rand010 = RandBBSfloat(seed010) * 2 - 1;
	float rand110 = RandBBSfloat(seed110) * 2 - 1;
	float rand001 = RandBBSfloat(seed001) * 2 - 1;
	float rand101 = RandBBSfloat(seed101) * 2 - 1;
	float rand011 = RandBBSfloat(seed011) * 2 - 1;
	float rand111 = RandBBSfloat(seed111) * 2 - 1;

	float3 Weights = PerlinRamp(float4(fv, 0)).xyz;

	float i = lerp(lerp(rand000, rand100, Weights.x), lerp(rand010, rand110, Weights.x), Weights.y);
	float j = lerp(lerp(rand001, rand101, Weights.x), lerp(rand011, rand111, Weights.x), Weights.y);
	return lerp(i, j, Weights.z).x;
}









float GradientNoise3D_TEX(float3 v, bool bTiling, float RepeatSize)
{
	bTiling = true;
	float3 fv = frac(v);
	float3 iv0 = NoiseTileWrap(floor(v), bTiling, RepeatSize);
	float3 iv1 = NoiseTileWrap(iv0 + 1, bTiling, RepeatSize);

	const int2 ZShear = int2(17, 89);

	float2 OffsetA = iv0.z * ZShear;
	float2 OffsetB = OffsetA + ZShear;
	if (bTiling)
	{
		OffsetB = iv1.z * ZShear;
	}


	float ts = 1 / 128.0f;


	float2 TexA0 = (iv0.xy + OffsetA + 0.5f) * ts;
	float2 TexB0 = (iv0.xy + OffsetB + 0.5f) * ts;


	float2 TexA1 = TexA0 + ts;
	float2 TexB1 = TexB0 + ts;
	if (bTiling)
	{
		TexA1 = (iv1.xy + OffsetA + 0.5f) * ts;
		TexB1 = (iv1.xy + OffsetB + 0.5f) * ts;
	}



	float3 A = Texture2DSampleLevel(View_PerlinNoiseGradientTexture, View_PerlinNoiseGradientTextureSampler, float2(TexA0.x, TexA0.y), 0).xyz * 2 - 1;
	float3 B = Texture2DSampleLevel(View_PerlinNoiseGradientTexture, View_PerlinNoiseGradientTextureSampler, float2(TexA1.x, TexA0.y), 0).xyz * 2 - 1;
	float3 C = Texture2DSampleLevel(View_PerlinNoiseGradientTexture, View_PerlinNoiseGradientTextureSampler, float2(TexA0.x, TexA1.y), 0).xyz * 2 - 1;
	float3 D = Texture2DSampleLevel(View_PerlinNoiseGradientTexture, View_PerlinNoiseGradientTextureSampler, float2(TexA1.x, TexA1.y), 0).xyz * 2 - 1;
	float3 E = Texture2DSampleLevel(View_PerlinNoiseGradientTexture, View_PerlinNoiseGradientTextureSampler, float2(TexB0.x, TexB0.y), 0).xyz * 2 - 1;
	float3 F = Texture2DSampleLevel(View_PerlinNoiseGradientTexture, View_PerlinNoiseGradientTextureSampler, float2(TexB1.x, TexB0.y), 0).xyz * 2 - 1;
	float3 G = Texture2DSampleLevel(View_PerlinNoiseGradientTexture, View_PerlinNoiseGradientTextureSampler, float2(TexB0.x, TexB1.y), 0).xyz * 2 - 1;
	float3 H = Texture2DSampleLevel(View_PerlinNoiseGradientTexture, View_PerlinNoiseGradientTextureSampler, float2(TexB1.x, TexB1.y), 0).xyz * 2 - 1;

	float a = dot(A, fv - float3(0, 0, 0));
	float b = dot(B, fv - float3(1, 0, 0));
	float c = dot(C, fv - float3(0, 1, 0));
	float d = dot(D, fv - float3(1, 1, 0));
	float e = dot(E, fv - float3(0, 0, 1));
	float f = dot(F, fv - float3(1, 0, 1));
	float g = dot(G, fv - float3(0, 1, 1));
	float h = dot(H, fv - float3(1, 1, 1));

	float3 Weights = PerlinRamp(frac(float4(fv, 0))).xyz;

	float i = lerp(lerp(a, b, Weights.x), lerp(c, d, Weights.x), Weights.y);
	float j = lerp(lerp(e, f, Weights.x), lerp(g, h, Weights.x), Weights.y);

	return lerp(i, j, Weights.z);
}



float FastGradientPerlinNoise3D_TEX(float3 xyz)
{

	float Extent = 16;



	xyz = frac(xyz / (Extent - 1)) * (Extent - 1);


	float3 uvw = frac(xyz);


	float3 p0 = xyz - uvw;


	float3 f = PerlinRamp(float4(uvw, 0)).xyz;

	float3 p = p0 + f;

	float4 NoiseSample = Texture3DSampleLevel(View_PerlinNoise3DTexture, View_PerlinNoise3DTextureSampler, p / Extent + 0.5f / Extent, 0);



	float3 n = NoiseSample.xyz * 255.0f / 127.0f - 1.0f;
	float d = NoiseSample.w * 255.f - 127;
	return dot(xyz, n) - d;
}





float3 VoronoiCornerSample(float3 pos, int Quality)
{

	float3 noise = float3(Rand3DPCG16(int3(pos))) / 0xffff - 0.5;



	if (Quality <= 2)
	{
		return normalize(noise) * 0.2588;
	}



	if (Quality == 3)
	{
		return normalize(noise) * 0.3090;
	}


	return noise;
}








float4 VoronoiCompare(float4 minval, float3 candidate, float3 offset, bool bDistanceOnly)
{
	if (bDistanceOnly)
	{
		return float4(0, 0, 0, min(minval.w, dot(offset, offset)));
	}
	else
	{
		float newdist = dot(offset, offset);
		return newdist > minval.w ? minval : float4(candidate, newdist);
	}
}


float4 VoronoiNoise3D_ALU(float3 v, int Quality, bool bTiling, float RepeatSize, bool bDistanceOnly)
{
	float3 fv = frac(v), fv2 = frac(v + 0.5);
	float3 iv = floor(v), iv2 = floor(v + 0.5);


	float4 mindist = float4(0,0,0,100);
	float3 p, offset;


	if (Quality == 3)
	{
		[unroll(3)]  for (offset.x = -1; offset.x <= 1; ++offset.x)
		{
			[unroll(3)]  for (offset.y = -1; offset.y <= 1; ++offset.y)
			{
				[unroll(3)]  for (offset.z = -1; offset.z <= 1; ++offset.z)
				{
					p = offset + VoronoiCornerSample(NoiseTileWrap(iv2 + offset, bTiling, RepeatSize), Quality);
					mindist = VoronoiCompare(mindist, iv2 + p, fv2 - p, bDistanceOnly);
				}
			}
		}
	}


	else
	{
		[unroll(2)]  for (offset.x = 0; offset.x <= 1; ++offset.x)
		{
			[unroll(2)]  for (offset.y = 0; offset.y <= 1; ++offset.y)
			{
				[unroll(2)]  for (offset.z = 0; offset.z <= 1; ++offset.z)
				{
					p = offset + VoronoiCornerSample(NoiseTileWrap(iv + offset, bTiling, RepeatSize), Quality);
					mindist = VoronoiCompare(mindist, iv + p, fv - p, bDistanceOnly);


					if (Quality == 2)
					{

						p = offset + VoronoiCornerSample(NoiseTileWrap(iv2 + offset, bTiling, RepeatSize) + 467, Quality);
						mindist = VoronoiCompare(mindist, iv2 + p, fv2 - p, bDistanceOnly);
					}
				}
			}
		}
	}


	if (Quality >= 4)
	{
		[unroll(2)]  for (offset.x = -1; offset.x <= 2; offset.x += 3)
		{
			[unroll(2)]  for (offset.y = 0; offset.y <= 1; ++offset.y)
			{
				[unroll(2)]  for (offset.z = 0; offset.z <= 1; ++offset.z)
				{

					p = offset.xyz + VoronoiCornerSample(NoiseTileWrap(iv + offset.xyz, bTiling, RepeatSize), Quality);
					mindist = VoronoiCompare(mindist, iv + p, fv - p, bDistanceOnly);


					p = offset.yzx + VoronoiCornerSample(NoiseTileWrap(iv + offset.yzx, bTiling, RepeatSize), Quality);
					mindist = VoronoiCompare(mindist, iv + p, fv - p, bDistanceOnly);


					p = offset.zxy + VoronoiCornerSample(NoiseTileWrap(iv + offset.zxy, bTiling, RepeatSize), Quality);
					mindist = VoronoiCompare(mindist, iv + p, fv - p, bDistanceOnly);
				}
			}
		}
	}


	return float4(mindist.xyz, sqrt(mindist.w));
}







float3 ComputeSimplexWeights2D(float2 OrthogonalPos, out float2 PosA, out float2 PosB, out float2 PosC)
{
	float2 OrthogonalPosFloor = floor(OrthogonalPos);
	PosA = OrthogonalPosFloor;
	PosB = PosA + float2(1, 1);

	float2 LocalPos = OrthogonalPos - OrthogonalPosFloor;

	PosC = PosA + ((LocalPos.x > LocalPos.y) ? float2(1,0) : float2(0,1));

	float b = min(LocalPos.x, LocalPos.y);
	float c = abs(LocalPos.y - LocalPos.x);
	float a = 1.0f - b - c;

	return float3(a, b, c);
}



float4 ComputeSimplexWeights3D(float3 OrthogonalPos, out float3 PosA, out float3 PosB, out float3 PosC, out float3 PosD)
{
	float3 OrthogonalPosFloor = floor(OrthogonalPos);

	PosA = OrthogonalPosFloor;
	PosB = PosA + float3(1, 1, 1);

	OrthogonalPos -= OrthogonalPosFloor;

	float Largest = max(OrthogonalPos.x, max(OrthogonalPos.y, OrthogonalPos.z));
	float Smallest = min(OrthogonalPos.x, min(OrthogonalPos.y, OrthogonalPos.z));

	PosC = PosA + float3(Largest == OrthogonalPos.x, Largest == OrthogonalPos.y, Largest == OrthogonalPos.z);
	PosD = PosA + float3(Smallest != OrthogonalPos.x, Smallest != OrthogonalPos.y, Smallest != OrthogonalPos.z);

	float4 ret;

	float RG = OrthogonalPos.x - OrthogonalPos.y;
	float RB = OrthogonalPos.x - OrthogonalPos.z;
	float GB = OrthogonalPos.y - OrthogonalPos.z;

	ret.b =
		  min(max(0, RG), max(0, RB))
		+ min(max(0, -RG), max(0, GB))
		+ min(max(0, -RB), max(0, -GB));

	ret.a =
		  min(max(0, -RG), max(0, -RB))
		+ min(max(0, RG), max(0, -GB))
		+ min(max(0, RB), max(0, GB));

	ret.g = Smallest;
	ret.r = 1.0f - ret.g - ret.b - ret.a;

	return ret;
}

float2 GetPerlinNoiseGradientTextureAt(float2 v)
{
	float2 TexA = (v.xy + 0.5f) / 128.0f;


	float3 p = Texture2DSampleLevel(View_PerlinNoiseGradientTexture, View_PerlinNoiseGradientTextureSampler, TexA, 0).xyz * 2 - 1;
	return normalize(p.xy + p.z * 0.33f);
}

float3 GetPerlinNoiseGradientTextureAt(float3 v)
{
	const float2 ZShear = float2(17.0f, 89.0f);

	float2 OffsetA = v.z * ZShear;
	float2 TexA = (v.xy + OffsetA + 0.5f) / 128.0f;

	return Texture2DSampleLevel(View_PerlinNoiseGradientTexture, View_PerlinNoiseGradientTextureSampler, TexA , 0).xyz * 2 - 1;
}

float2 SkewSimplex(float2 In)
{
	return In + dot(In, (sqrt(3.0f) - 1.0f) * 0.5f );
}
float2 UnSkewSimplex(float2 In)
{
	return In - dot(In, (3.0f - sqrt(3.0f)) / 6.0f );
}
float3 SkewSimplex(float3 In)
{
	return In + dot(In, 1.0 / 3.0f );
}
float3 UnSkewSimplex(float3 In)
{
	return In - dot(In, 1.0 / 6.0f );
}




float GradientSimplexNoise2D_TEX(float2 EvalPos)
{
	float2 OrthogonalPos = SkewSimplex(EvalPos);

	float2 PosA, PosB, PosC, PosD;
	float3 Weights = ComputeSimplexWeights2D(OrthogonalPos, PosA, PosB, PosC);


	float2 A = GetPerlinNoiseGradientTextureAt(PosA);
	float2 B = GetPerlinNoiseGradientTextureAt(PosB);
	float2 C = GetPerlinNoiseGradientTextureAt(PosC);

	PosA = UnSkewSimplex(PosA);
	PosB = UnSkewSimplex(PosB);
	PosC = UnSkewSimplex(PosC);

	float DistanceWeight;

	DistanceWeight = saturate(0.5f - length2(EvalPos - PosA)); DistanceWeight *= DistanceWeight; DistanceWeight *= DistanceWeight;
	float a = dot(A, EvalPos - PosA) * DistanceWeight;
	DistanceWeight = saturate(0.5f - length2(EvalPos - PosB)); DistanceWeight *= DistanceWeight; DistanceWeight *= DistanceWeight;
	float b = dot(B, EvalPos - PosB) * DistanceWeight;
	DistanceWeight = saturate(0.5f - length2(EvalPos - PosC)); DistanceWeight *= DistanceWeight; DistanceWeight *= DistanceWeight;
	float c = dot(C, EvalPos - PosC) * DistanceWeight;

	return 70 * (a + b + c);
}






float SimplexNoise3D_TEX(float3 EvalPos)
{
	float3 OrthogonalPos = SkewSimplex(EvalPos);

	float3 PosA, PosB, PosC, PosD;
	float4 Weights = ComputeSimplexWeights3D(OrthogonalPos, PosA, PosB, PosC, PosD);


	float3 A = GetPerlinNoiseGradientTextureAt(PosA);
	float3 B = GetPerlinNoiseGradientTextureAt(PosB);
	float3 C = GetPerlinNoiseGradientTextureAt(PosC);
	float3 D = GetPerlinNoiseGradientTextureAt(PosD);

	PosA = UnSkewSimplex(PosA);
	PosB = UnSkewSimplex(PosB);
	PosC = UnSkewSimplex(PosC);
	PosD = UnSkewSimplex(PosD);

	float DistanceWeight;

	DistanceWeight = saturate(0.6f - length2(EvalPos - PosA)); DistanceWeight *= DistanceWeight; DistanceWeight *= DistanceWeight;
	float a = dot(A, EvalPos - PosA) * DistanceWeight;
	DistanceWeight = saturate(0.6f - length2(EvalPos - PosB)); DistanceWeight *= DistanceWeight; DistanceWeight *= DistanceWeight;
	float b = dot(B, EvalPos - PosB) * DistanceWeight;
	DistanceWeight = saturate(0.6f - length2(EvalPos - PosC)); DistanceWeight *= DistanceWeight; DistanceWeight *= DistanceWeight;
	float c = dot(C, EvalPos - PosC) * DistanceWeight;
	DistanceWeight = saturate(0.6f - length2(EvalPos - PosD)); DistanceWeight *= DistanceWeight; DistanceWeight *= DistanceWeight;
	float d = dot(D, EvalPos - PosD) * DistanceWeight;

	return 32 * (a + b + c + d);
}


float VolumeRaymarch(float3 posPixelWS, float3 posCameraWS)
{
	float ret = 0;
	int cnt = 60;

	[loop]  for(int i=0; i < cnt; ++i)
	{
		ret += saturate(FastGradientPerlinNoise3D_TEX(lerp(posPixelWS, posCameraWS, i/(float)cnt) * 0.01) - 0.2f);
	}

	return ret / cnt * (length(posPixelWS - posCameraWS) * 0.001f );
}
#line 756 "/Engine/Private/Common.ush"
#line 761 "/Engine/Private/Common.ush"
float  PhongShadingPow( float  X,  float  Y)
{
#line 779 "/Engine/Private/Common.ush"
	return ClampedPow(X, Y);
}
#line 801 "/Engine/Private/Common.ush"
Texture2D LightAttenuationTexture;
SamplerState LightAttenuationTextureSampler;





float ConvertTangentUnormToSnorm8(float Input)
{
	int IntVal = int(round(Input * 255.0f));

	IntVal = (IntVal > 127) ? (IntVal | 0xFFFFFF80) : IntVal;
	return clamp(IntVal / 127.0f, -1, 1);
}

float2 ConvertTangentUnormToSnorm8(float2 Input)
{
	int2 IntVal = int2(round(Input * 255.0f));

	IntVal = (IntVal > 127) ? (IntVal | 0xFFFFFF80) : IntVal;
	return clamp(IntVal / 127.0f, -1, 1);
}

float3 ConvertTangentUnormToSnorm8(float3 Input)
{
	int3 IntVal = int3(round(Input * 255.0f));
	IntVal = (IntVal > 127) ? (IntVal | 0xFFFFFF80) : IntVal;
	return clamp(IntVal / 127.0f, -1, 1);
}

float4 ConvertTangentUnormToSnorm8(float4 Input)
{
	int4 IntVal = int4(round(Input * 255.0f));

	IntVal = (IntVal > 127) ? (IntVal | 0xFFFFFF80) : IntVal;
	return clamp(IntVal / 127.0f, -1, 1);
}

float ConvertTangentUnormToSnorm16(float Input)
{
	int IntVal = int(round(Input * 65535.0f));

	IntVal = (IntVal > 32767) ? (IntVal | 0xFFFF8000) : IntVal;
	return clamp(IntVal / 32767.0f, -1, 1);
}

float2 ConvertTangentUnormToSnorm16(float2 Input)
{
	int2 IntVal = int2(round(Input * 65535.0f));

	IntVal = (IntVal > 32767) ? (IntVal | 0xFFFFFF80) : IntVal;
	return clamp(IntVal / 32767.0f, -1, 1);
}

float3 ConvertTangentUnormToSnorm16(float3 Input)
{
	int3 IntVal = int3(round(Input * 65535.0f));
	IntVal = (IntVal > 32767) ? (IntVal | 0xFFFFFF80) : IntVal;
	return clamp(IntVal / 32767.0f, -1, 1);
}

float4 ConvertTangentUnormToSnorm16(float4 Input)
{
	int4 IntVal = int4(round(Input * 65535.0f));

	IntVal = (IntVal > 32767) ? (IntVal | 0xFFFFFF80) : IntVal;
	return clamp(IntVal / 32767.0f, -1, 1);
}

float ConvertTangentSnormToUnorm8(float Input)
{
	float Res = Input >= 0.0f ? Input * 127 : ((Input + 1.0) * 127) + 128;
	return clamp(Res / 255, 0.0f, 0.99f);
}

float2 ConvertTangentSnormToUnorm8(float2 Input)
{
	float2 Res = Input >= 0.0f ? Input * 127 : ((Input + 1.0) * 127) + 128;
	return clamp(Res / 255, 0.0f, 0.99f);
}

float3 ConvertTangentSnormToUnorm8(float3 Input)
{
	float3 Res = Input >= 0.0f ? Input * 127 : ((Input + 1.0) * 127) + 128;
	return clamp(Res / 255, 0.0f, 0.99f);
}

float4 ConvertTangentSnormToUnorm8(float4 Input)
{
	float4 Res = Input >= 0.0f ? Input * 127 : ((Input + 1.0) * 127) + 128;
	return clamp(Res / 255, 0.0f, 0.99f);
}

float ConvertTangentSnormToUnorm16(float Input)
{
	float Res = Input >= 0.0f ? Input * 32767 : ((Input + 1.0) * 32767) + 32768;
	return clamp(Res / 65535, 0.0f, 0.99f);
}

float2 ConvertTangentSnormToUnorm16(float2 Input)
{
	float2 Res = Input >= 0.0f ? Input * 32767 : ((Input + 1.0) * 32767) + 32768;
	return clamp(Res / 65535, 0.0f, 0.99f);
}

float3 ConvertTangentSnormToUnorm16(float3 Input)
{
	float3 Res = Input >= 0.0f ? Input * 32767 : ((Input + 1.0) * 32767) + 32768;
	return clamp(Res / 65535, 0.0f, 0.99f);
}

float4 ConvertTangentSnormToUnorm16(float4 Input)
{
	float4 Res = Input >= 0.0f ? Input * 32767 : ((Input + 1.0) * 32767) + 32768;
	return clamp(Res / 65535, 0.0f, 0.99f);
}






uint PackUnorm2x16(float2 v)
{
	uint2 sv = uint2(round(clamp(v, 0.0, 1.0) * 65535.0));
	return (sv.x | (sv.y << 16u));
}

uint PackSnorm2x16(float2 v)
{
	uint2 sv = uint2(round(clamp(v, -1.0, 1.0) * 32767.0) + 32767.0);
	return (sv.x | (sv.y << 16u));
}

float2 UnpackUnorm2x16(uint p)
{
	float2 Ret;
	Ret.x = (p & 0xffff) * rcp(65535.0f);
	Ret.y = (p >> 16u) * rcp(65535.0f);
	return Ret;
}

float2 UnpackSnorm2x16(uint p)
{
	float2 Ret;
	Ret.x = clamp((float(p & 0xffff) - 32767.0f) * rcp(32767.0f), -1.0, 1.0);
	Ret.y = clamp((float(p >> 16u) - 32767.0f) * rcp(32767.0f), -1.0, 1.0);
	return Ret;
}

float Square( float x )
{
	return x*x;
}

float2 Square( float2 x )
{
	return x*x;
}

float3 Square( float3 x )
{
	return x*x;
}

float4 Square( float4 x )
{
	return x*x;
}

float Pow2( float x )
{
	return x*x;
}

float2 Pow2( float2 x )
{
	return x*x;
}

float3 Pow2( float3 x )
{
	return x*x;
}

float4 Pow2( float4 x )
{
	return x*x;
}

float Pow3( float x )
{
	return x*x*x;
}

float2 Pow3( float2 x )
{
	return x*x*x;
}

float3 Pow3( float3 x )
{
	return x*x*x;
}

float4 Pow3( float4 x )
{
	return x*x*x;
}

float Pow4( float x )
{
	float xx = x*x;
	return xx * xx;
}

float2 Pow4( float2 x )
{
	float2 xx = x*x;
	return xx * xx;
}

float3 Pow4( float3 x )
{
	float3 xx = x*x;
	return xx * xx;
}

float4 Pow4( float4 x )
{
	float4 xx = x*x;
	return xx * xx;
}

float Pow5( float x )
{
	float xx = x*x;
	return xx * xx * x;
}

float2 Pow5( float2 x )
{
	float2 xx = x*x;
	return xx * xx * x;
}

float3 Pow5( float3 x )
{
	float3 xx = x*x;
	return xx * xx * x;
}

float4 Pow5( float4 x )
{
	float4 xx = x*x;
	return xx * xx * x;
}

float Pow6( float x )
{
	float xx = x*x;
	return xx * xx * xx;
}

float2 Pow6( float2 x )
{
	float2 xx = x*x;
	return xx * xx * xx;
}

float3 Pow6( float3 x )
{
	float3 xx = x*x;
	return xx * xx * xx;
}

float4 Pow6( float4 x )
{
	float4 xx = x*x;
	return xx * xx * xx;
}


float  AtanFast(  float  x )
{

	float3  A = x < 1 ?  float3 ( x, 0, 1 ) :  float3 ( 1/x, 0.5 * PI, -1 );
	return A.y + A.z * ( ( ( -0.130234 * A.x - 0.0954105 ) * A.x + 1.00712 ) * A.x - 0.00001203333 );
}


float  EncodeLightAttenuation( float  InColor)
{


	return sqrt(InColor);
}


float4  EncodeLightAttenuation( float4  InColor)
{
	return sqrt(InColor);
}


float  DecodeLightAttenuation( float  InColor)
{
	return Square(InColor);
}


float4  DecodeLightAttenuation( float4  InColor)
{
	return Square(InColor);
}


float4  RGBTEncode( float3  Color)
{
	float4  RGBT;
	float  Max = max(max(Color.r, Color.g), max(Color.b, 1e-6));
	float  RcpMax = rcp(Max);
	RGBT.rgb = Color.rgb * RcpMax;
	RGBT.a = Max * rcp(1.0 + Max);
	return RGBT;
}

float3  RGBTDecode( float4  RGBT)
{
	RGBT.a = RGBT.a * rcp(1.0 - RGBT.a);
	return RGBT.rgb * RGBT.a;
}



float4  RGBMEncode(  float3  Color )
{
	Color *= 1.0 / 64.0;

	float4 rgbm;
	rgbm.a = saturate( max( max( Color.r, Color.g ), max( Color.b, 1e-6 ) ) );
	rgbm.a = ceil( rgbm.a * 255.0 ) / 255.0;
	rgbm.rgb = Color / rgbm.a;
	return rgbm;
}

float4  RGBMEncodeFast(  float3  Color )
{

	float4  rgbm;
	rgbm.a = dot( Color, 255.0 / 64.0 );
	rgbm.a = ceil( rgbm.a );
	rgbm.rgb = Color / rgbm.a;
	rgbm *=  float4 ( 255.0 / 64.0, 255.0 / 64.0, 255.0 / 64.0, 1.0 / 255.0 );
	return rgbm;
}

float3  RGBMDecode(  float4  rgbm,  float  MaxValue )
{
	return rgbm.rgb * (rgbm.a * MaxValue);
}

float3  RGBMDecode(  float4  rgbm )
{
	return rgbm.rgb * (rgbm.a * 64.0f);
}

float4  RGBTEncode8BPC( float3  Color,  float  Range)
{
	float  Max = max(max(Color.r, Color.g), max(Color.b, 1e-6));
	Max = min(Max, Range);

	float4  RGBT;
	RGBT.a = (Range + 1) / Range * Max / (1 + Max);


	RGBT.a = ceil(RGBT.a*255.0) / 255.0;
	Max = RGBT.a / (1 + 1 / Range - RGBT.a);

	float  RcpMax = rcp(Max);
	RGBT.rgb = Color.rgb * RcpMax;
	return RGBT;
}

float3  RGBTDecode8BPC( float4  RGBT,  float  Range)
{
	RGBT.a = RGBT.a / (1 + 1 / Range - RGBT.a);
	return RGBT.rgb * RGBT.a;
}
#line 1208 "/Engine/Private/Common.ush"
float2 CalcScreenUVFromOffsetFraction(float4 ScreenPosition, float2 OffsetFraction)
{
	float2 NDC = ScreenPosition.xy / ScreenPosition.w;



	float2 OffsetNDC = clamp(NDC + OffsetFraction * float2(2, -2), -.999f, .999f);
	return float2(OffsetNDC * ResolvedView.ScreenPositionScaleBias.xy + ResolvedView.ScreenPositionScaleBias.wz);
}

float4 GetPerPixelLightAttenuation(float2 UV)
{
	return DecodeLightAttenuation(Texture2DSampleLevel(LightAttenuationTexture, LightAttenuationTextureSampler, UV, 0));
}




float ConvertFromDeviceZ(float DeviceZ)
{

	return DeviceZ * View_InvDeviceZToWorldZTransform[0] + View_InvDeviceZToWorldZTransform[1] + 1.0f / (DeviceZ * View_InvDeviceZToWorldZTransform[2] - View_InvDeviceZToWorldZTransform[3]);
}




float ConvertToDeviceZ(float SceneDepth)
{
	[flatten]
	if (View_ViewToClip[3][3] < 1.0f)
	{

		return 1.0f / ((SceneDepth + View_InvDeviceZToWorldZTransform[3]) * View_InvDeviceZToWorldZTransform[2]);
	}
	else
	{

		return SceneDepth * View_ViewToClip[2][2] + View_ViewToClip[3][2];
	}
}

float2 ScreenPositionToBufferUV(float4 ScreenPosition)
{
	return float2(ScreenPosition.xy / ScreenPosition.w * ResolvedView.ScreenPositionScaleBias.xy + ResolvedView.ScreenPositionScaleBias.wz);
}

float2 SvPositionToBufferUV(float4 SvPosition)
{
	return SvPosition.xy * View_BufferSizeAndInvSize.zw;
}


float3 SvPositionToTranslatedWorld(float4 SvPosition)
{
	float4 HomWorldPos = mul(float4(SvPosition.xyz, 1), View_SVPositionToTranslatedWorld);

	return HomWorldPos.xyz / HomWorldPos.w;
}


float3 SvPositionToResolvedTranslatedWorld(float4 SvPosition)
{
	float4 HomWorldPos = mul(float4(SvPosition.xyz, 1), ResolvedView.SVPositionToTranslatedWorld);

	return HomWorldPos.xyz / HomWorldPos.w;
}


FLWCVector3 SvPositionToWorld(float4 SvPosition)
{
	float3 TranslatedWorldPosition = SvPositionToTranslatedWorld(SvPosition);
	return LWCSubtract(TranslatedWorldPosition,  GetPrimaryView() .PreViewTranslation);
}


float4 SvPositionToScreenPosition(float4 SvPosition)
{



	float2 PixelPos = SvPosition.xy - View_ViewRectMin.xy;


	float3 NDCPos = float3( (PixelPos * View_ViewSizeAndInvSize.zw - 0.5f) * float2(2, -2), SvPosition.z);


	return float4(NDCPos.xyz, 1) * SvPosition.w;
}


float4 SvPositionToResolvedScreenPosition(float4 SvPosition)
{
	float2 PixelPos = SvPosition.xy - ResolvedView.ViewRectMin.xy;


	float3 NDCPos = float3( (PixelPos * ResolvedView.ViewSizeAndInvSize.zw - 0.5f) * float2(2, -2), SvPosition.z);


	return float4(NDCPos.xyz, 1) * SvPosition.w;
}

void SvPositionToResolvedScreenPositionDeriv(float4 SvPosition, float2 PPZ_DDX_DDY, float2 W_DDX_DDY, inout float4 ScreenPosition, inout float4 ScreenPositionDDX, inout float4 ScreenPositionDDY)
{
	float2 PixelPos = SvPosition.xy - ResolvedView.ViewRectMin.xy;


	float4 NDCPos = float4((PixelPos * ResolvedView.ViewSizeAndInvSize.zw - 0.5f) * float2(2, -2), SvPosition.z, 1.0f);
	float4 NDCPosDDX = float4(ResolvedView.ViewSizeAndInvSize.z * 2.0f, 0.0f, PPZ_DDX_DDY.x, 0.0f);
	float4 NDCPosDDY = float4(ResolvedView.ViewSizeAndInvSize.w * 2.0f, 0.0f, PPZ_DDX_DDY.y, 0.0f);

	ScreenPosition = NDCPos * SvPosition.w;
	ScreenPositionDDX = NDCPos * W_DDX_DDY.x + NDCPosDDX * SvPosition.w;
	ScreenPositionDDY = NDCPos * W_DDX_DDY.y + NDCPosDDY * SvPosition.w;
}

float2 SvPositionToViewportUV(float4 SvPosition)
{

	float2 PixelPos = SvPosition.xy - View_ViewRectMin.xy;

	return PixelPos.xy * View_ViewSizeAndInvSize.zw;
}

float2 BufferUVToViewportUV(float2 BufferUV)
{
	float2 PixelPos = BufferUV.xy * View_BufferSizeAndInvSize.xy - View_ViewRectMin.xy;
	return PixelPos.xy * View_ViewSizeAndInvSize.zw;
}

float2 ViewportUVToBufferUV(float2 ViewportUV)
{
	float2 PixelPos = ViewportUV * View_ViewSizeAndInvSize.xy;
	return (PixelPos + View_ViewRectMin.xy) * View_BufferSizeAndInvSize.zw;
}


float2 ViewportUVToScreenPos(float2 ViewportUV)
{
	return float2(2 * ViewportUV.x - 1, 1 - 2 * ViewportUV.y);
}

float2 ScreenPosToViewportUV(float2 ScreenPos)
{
	return float2(0.5 + 0.5 * ScreenPos.x, 0.5 - 0.5 * ScreenPos.y);
}



float3 ScreenToViewPos(float2 ViewportUV, float SceneDepth)
{
	float2 ProjViewPos;

	ProjViewPos.x = ViewportUV.x * View_ScreenToViewSpace.x + View_ScreenToViewSpace.z;
	ProjViewPos.y = ViewportUV.y * View_ScreenToViewSpace.y + View_ScreenToViewSpace.w;
	return float3(ProjViewPos * SceneDepth, SceneDepth);
}
#line 1372 "/Engine/Private/Common.ush"
float2  ScreenAlignedPosition( float4 ScreenPosition )
{
	return  float2 (ScreenPositionToBufferUV(ScreenPosition));
}
#line 1380 "/Engine/Private/Common.ush"
float2  ScreenAlignedUV(  float2  UV )
{
	return (UV* float2 (2,-2) +  float2 (-1,1))*View_ScreenPositionScaleBias.xy + View_ScreenPositionScaleBias.wz;
}
#line 1388 "/Engine/Private/Common.ush"
float2  GetViewportCoordinates( float2  InFragmentCoordinates)
{
	return InFragmentCoordinates;
}
#line 1396 "/Engine/Private/Common.ush"
float4  UnpackNormalMap(  float4  TextureSample )
{



		float2  NormalXY = TextureSample.rg;


	NormalXY = NormalXY *  float2 (2.0f,2.0f) -  float2 (1.0f,1.0f);
	float  NormalZ = sqrt( saturate( 1.0f - dot( NormalXY, NormalXY ) ) );
	return  float4 ( NormalXY.xy, NormalZ, 1.0f );
}


float AntialiasedTextureMask( Texture2D Tex, SamplerState Sampler, float2 UV, float ThresholdConst, int Channel )
{

	float4  MaskConst =  float4 (Channel == 0, Channel == 1, Channel == 2, Channel == 3);


	const float WidthConst = 1.0f;
	float InvWidthConst = 1 / WidthConst;
#line 1440 "/Engine/Private/Common.ush"
	float Result;
	{

		float Sample1 = dot(MaskConst, Texture2DSample(Tex, Sampler, UV));


		float2 TexDD = float2(DDX(Sample1), DDY(Sample1));

		float TexDDLength = max(abs(TexDD.x), abs(TexDD.y));
		float Top = InvWidthConst * (Sample1 - ThresholdConst);
		Result = Top / TexDDLength + ThresholdConst;
	}

	Result = saturate(Result);

	return Result;
}



float Noise3D_Multiplexer(int Function, float3 Position, int Quality, bool bTiling, float RepeatSize)
{

	switch(Function)
	{
		case 0:
			return SimplexNoise3D_TEX(Position);
		case 1:
			return GradientNoise3D_TEX(Position, bTiling, RepeatSize);
		case 2:
			return FastGradientPerlinNoise3D_TEX(Position);
		case 3:
			return GradientNoise3D_ALU(Position, bTiling, RepeatSize);
		case 4:
			return ValueNoise3D_ALU(Position, bTiling, RepeatSize);
		default:
			return VoronoiNoise3D_ALU(Position, Quality, bTiling, RepeatSize, true).w * 2. - 1.;
	}
	return 0;
}



float  MaterialExpressionNoise(float3 Position, float Scale, int Quality, int Function, bool bTurbulence, uint Levels, float OutputMin, float OutputMax, float LevelScale, float FilterWidth, bool bTiling, float RepeatSize)
{
	Position *= Scale;
	FilterWidth *= Scale;

	float Out = 0.0f;
	float OutScale = 1.0f;
	float InvLevelScale = 1.0f / LevelScale;

	[loop]  for(uint i = 0; i < Levels; ++i)
	{

		OutScale *= saturate(1.0 - FilterWidth);

		if(bTurbulence)
		{
			Out += abs(Noise3D_Multiplexer(Function, Position, Quality, bTiling, RepeatSize)) * OutScale;
		}
		else
		{
			Out += Noise3D_Multiplexer(Function, Position, Quality, bTiling, RepeatSize) * OutScale;
		}

		Position *= LevelScale;
		RepeatSize *= LevelScale;
		OutScale *= InvLevelScale;
		FilterWidth *= LevelScale;
	}

	if(!bTurbulence)
	{

		Out = Out * 0.5f + 0.5f;
	}


	return lerp(OutputMin, OutputMax, Out);
}





float4  MaterialExpressionVectorNoise( float3  Position, int Quality, int Function, bool bTiling, float TileSize)
{
	float4 result = float4(0,0,0,1);
	float3x4 Jacobian = JacobianSimplex_ALU(Position, bTiling, TileSize);


	switch (Function)
	{
	case 0:
		result.xyz = float3(Rand3DPCG16(int3(floor(NoiseTileWrap(Position, bTiling, TileSize))))) / 0xffff;
		break;
	case 1:
		result.xyz = float3(Jacobian[0].w, Jacobian[1].w, Jacobian[2].w);
		break;
	case 2:
		result = Jacobian[0];
		break;
	case 3:
		result.xyz = float3(Jacobian[2][1] - Jacobian[1][2], Jacobian[0][2] - Jacobian[2][0], Jacobian[1][0] - Jacobian[0][1]);
		break;
	default:
		result = VoronoiNoise3D_ALU(Position, Quality, bTiling, TileSize, false);
		break;
	}
	return result;
}
#line 1567 "/Engine/Private/Common.ush"
float2 LineBoxIntersect(float3 RayOrigin, float3 RayEnd, float3 BoxMin, float3 BoxMax)
{
	float3 InvRayDir = 1.0f / (RayEnd - RayOrigin);


	float3 FirstPlaneIntersections = (BoxMin - RayOrigin) * InvRayDir;

	float3 SecondPlaneIntersections = (BoxMax - RayOrigin) * InvRayDir;

	float3 ClosestPlaneIntersections = min(FirstPlaneIntersections, SecondPlaneIntersections);

	float3 FurthestPlaneIntersections = max(FirstPlaneIntersections, SecondPlaneIntersections);

	float2 BoxIntersections;

	BoxIntersections.x = max(ClosestPlaneIntersections.x, max(ClosestPlaneIntersections.y, ClosestPlaneIntersections.z));

	BoxIntersections.y = min(FurthestPlaneIntersections.x, min(FurthestPlaneIntersections.y, FurthestPlaneIntersections.z));

	return saturate(BoxIntersections);
}


float  ComputeDistanceFromBoxToPoint( float3  Mins,  float3  Maxs,  float3  InPoint)
{
	float3  DistancesToMin = InPoint < Mins ? abs(InPoint - Mins) : 0;
	float3  DistancesToMax = InPoint > Maxs ? abs(InPoint - Maxs) : 0;


	float  Distance = dot(DistancesToMin, 1);
	Distance += dot(DistancesToMax, 1);
	return Distance;
}


float  ComputeSquaredDistanceFromBoxToPoint( float3  BoxCenter,  float3  BoxExtent,  float3  InPoint)
{
	float3  AxisDistances = max(abs(InPoint - BoxCenter) - BoxExtent, 0);
	return dot(AxisDistances, AxisDistances);
}


float ComputeDistanceFromBoxToPointInside(float3 BoxCenter, float3 BoxExtent, float3 InPoint)
{
	float3 DistancesToMin = max(InPoint - BoxCenter + BoxExtent, 0);
	float3 DistancesToMax = max(BoxCenter + BoxExtent - InPoint, 0);
	float3 ClosestDistances = min(DistancesToMin, DistancesToMax);
	return min(ClosestDistances.x, min(ClosestDistances.y, ClosestDistances.z));
}

bool RayHitSphere(float3 RayOrigin, float3 UnitRayDirection, float3 SphereCenter, float SphereRadius)
{
	float3 ClosestPointOnRay = max(0, dot(SphereCenter - RayOrigin, UnitRayDirection)) * UnitRayDirection;
	float3 CenterToRay = RayOrigin + ClosestPointOnRay - SphereCenter;
	return dot(CenterToRay, CenterToRay) <= Square(SphereRadius);
}

bool RaySegmentHitSphere(float3 RayOrigin, float3 UnitRayDirection, float RayLength, float3 SphereCenter, float SphereRadius)
{
	float DistanceAlongRay = dot(SphereCenter - RayOrigin, UnitRayDirection);
	float3 ClosestPointOnRay = DistanceAlongRay * UnitRayDirection;
	float3 CenterToRay = RayOrigin + ClosestPointOnRay - SphereCenter;
	return dot(CenterToRay, CenterToRay) <= Square(SphereRadius) && DistanceAlongRay > -SphereRadius && DistanceAlongRay - SphereRadius < RayLength;
}
#line 1636 "/Engine/Private/Common.ush"
float2 RayIntersectSphere(float3 RayOrigin, float3 RayDirection, float4 Sphere)
{
	float3 LocalPosition = RayOrigin - Sphere.xyz;
	float LocalPositionSqr = dot(LocalPosition, LocalPosition);

	float3 QuadraticCoef;
	QuadraticCoef.x = dot(RayDirection, RayDirection);
	QuadraticCoef.y = 2 * dot(RayDirection, LocalPosition);
	QuadraticCoef.z = LocalPositionSqr - Sphere.w * Sphere.w;

	float Discriminant = QuadraticCoef.y * QuadraticCoef.y - 4 * QuadraticCoef.x * QuadraticCoef.z;

	float2 Intersections = -1;


	[flatten]
	if (Discriminant >= 0)
	{
		float SqrtDiscriminant = sqrt(Discriminant);
		Intersections = (-QuadraticCoef.y + float2(-1, 1) * SqrtDiscriminant) / (2 * QuadraticCoef.x);
	}

	return Intersections;
}


float3  TransformTangentVectorToWorld( float3x3  TangentToWorld,  float3  InTangentVector)
{


	return mul(InTangentVector, TangentToWorld);
}


float3  TransformWorldVectorToTangent( float3x3  TangentToWorld,  float3  InWorldVector)
{


	return mul(TangentToWorld, InWorldVector);
}

float3 TransformWorldVectorToView(float3 InTangentVector)
{

	return mul(InTangentVector, (float3x3)ResolvedView.TranslatedWorldToView);
}


float  GetBoxPushout( float3  Normal, float3  Extent)
{
	return dot(abs(Normal * Extent),  float3 (1.0f, 1.0f, 1.0f));
}


void GenerateCoordinateSystem(float3 ZAxis, out float3 XAxis, out float3 YAxis)
{
	if (abs(ZAxis.x) > abs(ZAxis.y))
	{
		float InverseLength = 1.0f / sqrt(dot(ZAxis.xz, ZAxis.xz));
		XAxis = float3(-ZAxis.z * InverseLength, 0.0f, ZAxis.x * InverseLength);
	}
	else
	{
		float InverseLength = 1.0f / sqrt(dot(ZAxis.yz, ZAxis.yz));
		XAxis = float3(0.0f, ZAxis.z * InverseLength, -ZAxis.y * InverseLength);
	}

	YAxis = cross(ZAxis, XAxis);
}
#line 1715 "/Engine/Private/Common.ush"
struct FScreenVertexOutput
{




	noperspective  float2  UV : TEXCOORD0;

	float4 Position : SV_POSITION;
};




float4 EncodeVelocityToTexture(float3 V)
{

		V.xy = sign(V.xy) * sqrt(abs(V.xy)) * (2.0 / sqrt(2.0));




	float4 EncodedV;
	EncodedV.xy = V.xy * (0.499f * 0.5f) + 32767.0f / 65535.0f;


		uint Vz = asuint(V.z);

		EncodedV.z = saturate(float((Vz >> 16) & 0xFFFF) * rcp(65535.0f) + (0.1 / 65535.0f));
		EncodedV.w = saturate(float((Vz >> 0) & 0xFFFF) * rcp(65535.0f) + (0.1 / 65535.0f));
#line 1749 "/Engine/Private/Common.ush"
	return EncodedV;
}

float3 DecodeVelocityFromTexture(float4 EncodedV)
{
	const float InvDiv = 1.0f / (0.499f * 0.5f);

	float3 V;
	V.xy = EncodedV.xy * InvDiv - 32767.0f / 65535.0f * InvDiv;


		V.z = asfloat((uint(round(EncodedV.z * 65535.0f)) << 16) | uint(round(EncodedV.w * 65535.0f)));
#line 1766 "/Engine/Private/Common.ush"
		V.xy = (V.xy * abs(V.xy)) * 0.5;


	return V;
}


bool GetGIReplaceState()
{



	return false;

}

bool GetRayTracingQualitySwitch()
{



	return false;

}

bool GetPathTracingQualitySwitch()
{



	return false;

}



bool GetRuntimeVirtualTextureOutputSwitch()
{



	return false;

}


struct FWriteToSliceGeometryOutput
{
	FScreenVertexOutput Vertex;
	uint LayerIndex : SV_RenderTargetArrayIndex;
};







void DrawRectangle(
	in float4 InPosition,
	in float2 InTexCoord,
	out float4 OutPosition,
	out float2 OutTexCoord)
{
	OutPosition = InPosition;
	OutPosition.xy = -1.0f + 2.0f * (DrawRectangleParameters_PosScaleBias.zw + (InPosition.xy * DrawRectangleParameters_PosScaleBias.xy)) * DrawRectangleParameters_InvTargetSizeAndTextureSize.xy;
	OutPosition.xy *= float2( 1, -1 );
	OutTexCoord.xy = (DrawRectangleParameters_UVScaleBias.zw + (InTexCoord.xy * DrawRectangleParameters_UVScaleBias.xy)) * DrawRectangleParameters_InvTargetSizeAndTextureSize.zw;
}


void DrawRectangle(
	in float4 InPosition,
	in float2 InTexCoord,
	out float4 OutPosition,
	out float4 OutUVAndScreenPos)
{
	DrawRectangle(InPosition, InTexCoord, OutPosition, OutUVAndScreenPos.xy);
	OutUVAndScreenPos.zw = OutPosition.xy;
}


void DrawRectangle(in float4 InPosition, out float4 OutPosition)
{
	OutPosition = InPosition;
	OutPosition.xy = -1.0f + 2.0f * (DrawRectangleParameters_PosScaleBias.zw + (InPosition.xy * DrawRectangleParameters_PosScaleBias.xy)) * DrawRectangleParameters_InvTargetSizeAndTextureSize.xy;
	OutPosition.xy *= float2( 1, -1 );
}
#line 1866 "/Engine/Private/Common.ush"
float SafeSaturate(float In) { return saturate(In);}
float2 SafeSaturate(float2 In) { return saturate(In);}
float3 SafeSaturate(float3 In) { return saturate(In);}
float4 SafeSaturate(float4 In) { return saturate(In);}
#line 1895 "/Engine/Private/Common.ush"
bool IsFinite(float In) { return (asuint(In) & 0x7F800000) != 0x7F800000; }bool IsPositiveFinite(float In) { return asuint(In) < 0x7F800000; }float MakeFinite(float In) { return !IsFinite(In)? 0 : In; }float MakePositiveFinite(float In) { return !IsPositiveFinite(In)? 0 : In; }
bool2 IsFinite(float2 In) { return (asuint(In) & 0x7F800000) != 0x7F800000; }bool2 IsPositiveFinite(float2 In) { return asuint(In) < 0x7F800000; }float2 MakeFinite(float2 In) { return !IsFinite(In)? 0 : In; }float2 MakePositiveFinite(float2 In) { return !IsPositiveFinite(In)? 0 : In; }
bool3 IsFinite(float3 In) { return (asuint(In) & 0x7F800000) != 0x7F800000; }bool3 IsPositiveFinite(float3 In) { return asuint(In) < 0x7F800000; }float3 MakeFinite(float3 In) { return !IsFinite(In)? 0 : In; }float3 MakePositiveFinite(float3 In) { return !IsPositiveFinite(In)? 0 : In; }
bool4 IsFinite(float4 In) { return (asuint(In) & 0x7F800000) != 0x7F800000; }bool4 IsPositiveFinite(float4 In) { return asuint(In) < 0x7F800000; }float4 MakeFinite(float4 In) { return !IsFinite(In)? 0 : In; }float4 MakePositiveFinite(float4 In) { return !IsPositiveFinite(In)? 0 : In; }





bool GetShadowReplaceState()
{



	return false;

}

bool GetReflectionCapturePassSwitchState()
{
	return View_RenderingReflectionCaptureMask > 0.0f;
}

float IsShadowDepthShader()
{
	return GetShadowReplaceState() ? 1.0f : 0.0f;
}




float DecodePackedTwoChannelValue(float2 PackedHeight)
{
	return PackedHeight.x * 255.0 * 256.0 + PackedHeight.y * 255.0;
}

float DecodeHeightValue(float InValue)
{
	return (InValue - 32768.0) *  (1.0f/128.0f) ;
}

float DecodePackedHeight(float2 PackedHeight)
{
	return DecodeHeightValue(DecodePackedTwoChannelValue(PackedHeight));
}


uint ReverseBitsN(uint Bitfield, const uint BitCount)
{
	return reversebits(Bitfield) >> (32 - BitCount);
}


uint2 ZOrder2D(uint Index, const uint SizeLog2)
{
	uint2 Coord = 0;

	[unroll]
	for (uint i = 0; i < SizeLog2; i++)
	{
		Coord.x |= ((Index >> (2 * i + 0)) & 0x1) << i;
		Coord.y |= ((Index >> (2 * i + 1)) & 0x1) << i;
	}

	return Coord;
}

uint3 ZOrder3D(uint Index, const uint SizeLog2)
{
    uint3 Coord = 0;

    [unroll]
    for (uint i = 0; i < SizeLog2; i++)
    {
        Coord.x |= ((Index >> (3 * i + 0)) & 0x1) << i;
        Coord.y |= ((Index >> (3 * i + 1)) & 0x1) << i;
        Coord.z |= ((Index >> (3 * i + 2)) & 0x1) << i;
    }

    return Coord;
}

uint ZOrder3DEncode(uint3 Coord, const uint SizeLog2)
{
    uint Index = 0;

    [unroll]
    for (uint i = 0; i < SizeLog2; i++)
    {
        Index |= ((Coord.x >> i) & 0x1) << (3 * i + 0);
        Index |= ((Coord.y >> i) & 0x1) << (3 * i + 1);
        Index |= ((Coord.z >> i) & 0x1) << (3 * i + 2);
    }

    return Index;
}



struct FPixelShaderIn
{

	float4 SvPosition;


	uint Coverage;


	bool bIsFrontFace;
};

struct FPixelShaderOut
{

	float4 MRT[8];


	uint StrataOutput[3];


	uint Coverage;


	float Depth;
};
#line 2049 "/Engine/Private/Common.ush"
float4 GatherDepth(Texture2D Texture, float2 UV)
{

	float4 DeviceZ = Texture.GatherRed( D3DStaticBilinearClampedSampler , UV);

	return float4(
		ConvertFromDeviceZ(DeviceZ.x),
		ConvertFromDeviceZ(DeviceZ.y),
		ConvertFromDeviceZ(DeviceZ.z),
		ConvertFromDeviceZ(DeviceZ.w));
}
#line 10 "/Engine/Private/SceneTexturesCommon.ush"
#line 41 "/Engine/Private/SceneTexturesCommon.ush"
float3 CalcSceneColor(float2 ScreenUV)
{



	return Texture2DSampleLevel(SceneTexturesStruct_SceneColorTexture,  SceneTexturesStruct_PointClampSampler , ScreenUV, 0).rgb;

}

float4 CalcFullSceneColor(float2 ScreenUV)
{



	return Texture2DSample(SceneTexturesStruct_SceneColorTexture,  SceneTexturesStruct_PointClampSampler ,ScreenUV);

}


float CalcSceneDepth(float2 ScreenUV)
{



	return ConvertFromDeviceZ(Texture2DSampleLevel(SceneTexturesStruct_SceneDepthTexture,  SceneTexturesStruct_PointClampSampler , ScreenUV, 0).r);

}


float4 CalcSceneColorAndDepth( float2 ScreenUV )
{
	return float4(CalcSceneColor(ScreenUV), CalcSceneDepth(ScreenUV));
}


float LookupDeviceZ( float2 ScreenUV )
{




	return Texture2DSampleLevel(SceneTexturesStruct_SceneDepthTexture,  SceneTexturesStruct_PointClampSampler , ScreenUV, 0).r;

}


float LookupDeviceZ(uint2 PixelPos)
{



	return SceneTexturesStruct_SceneDepthTexture.Load(int3(PixelPos, 0)).r;

}


float CalcSceneDepth(uint2 PixelPos)
{



	float DeviceZ = SceneTexturesStruct_SceneDepthTexture.Load(int3(PixelPos, 0)).r;


	return ConvertFromDeviceZ(DeviceZ);

}


float4 GatherSceneDepth(float2 UV, float2 InvBufferSize)
{



	return GatherDepth(SceneTexturesStruct_SceneDepthTexture, UV);

}
#line 6 "/Engine/Private/SceneTextureParameters.ush"
#line 1 "DeferredShadingCommon.ush"
#line 9 "/Engine/Private/DeferredShadingCommon.ush"
#line 1 "ShadingCommon.ush"
#line 45 "/Engine/Private/ShadingCommon.ush"
float3 GetShadingModelColor(uint ShadingModelID)
{
#line 66 "/Engine/Private/ShadingCommon.ush"
	switch(ShadingModelID)
	{
		case  0 : return float3(0.1f, 0.1f, 0.2f);
		case  1 : return float3(0.1f, 1.0f, 0.1f);
		case  2 : return float3(1.0f, 0.1f, 0.1f);
		case  3 : return float3(0.6f, 0.4f, 0.1f);
		case  4 : return float3(0.1f, 0.4f, 0.4f);
		case  5 : return float3(0.2f, 0.6f, 0.5f);
		case  6 : return float3(0.2f, 0.2f, 0.8f);
		case  7 : return float3(0.6f, 0.1f, 0.5f);
		case  8 : return float3(0.7f, 1.0f, 1.0f);
		case  9 : return float3(0.3f, 1.0f, 1.0f);
		case  10 : return float3(0.5f, 0.5f, 1.0f);
		case  11 : return float3(1.0f, 0.8f, 0.3f);
		case  12 : return float3(0.8f, 0.2f, 0.65f);
		case  13 : return float3(1.0f, 1.0f, 0.0f);
		default: return float3(1.0f, 1.0f, 1.0f);
	}

}




bool GetShadingModelRequiresBackfaceLighting(uint ShadingModelID)
{
	return ShadingModelID ==  6 ;
}


float F0ToDielectricSpecular(float F0)
{
	return saturate(F0 / 0.08f);
}

float DielectricSpecularToF0(float Specular)
{
	return 0.08f * Specular;
}


float DielectricF0ToIor(float F0)
{
	return 2.0f / (1.0f - sqrt(F0)) - 1.0f;
}

float DielectricIorToF0(float Ior)
{
	const float F0Sqrt = (Ior-1)/(Ior+1);
	const float F0 = F0Sqrt*F0Sqrt;
	return F0;
}

float ComputeSpecularMicroOcclusion(float F0)
{

	return saturate(50.0 * F0);
}

float3 ComputeF0(float Specular, float3 BaseColor, float Metallic)
{
	return lerp(DielectricSpecularToF0(Specular).xxx, BaseColor, Metallic.xxx);
}

float3 ComputeF90(float3 F0, float3 EdgeColor, float Metallic)
{
	return lerp(1.0, EdgeColor, Metallic.xxx);
}

float MakeRoughnessSafe(float Roughness, float MinRoughness=0.001f)
{
	return clamp(Roughness, MinRoughness, 1.0f);
}

float ComputeHazyLobeRoughness(float Roughness, float Haziness)
{
	return lerp(Roughness, 1.0f, Haziness);
}
#line 10 "/Engine/Private/DeferredShadingCommon.ush"
#line 1 "LightAccumulator.ush"
#line 24 "/Engine/Private/LightAccumulator.ush"
struct FLightAccumulator
{
	float3 TotalLight;




	float ScatterableLightLuma;




	float3 ScatterableLight;



	float EstimatedCost;



	float3 TotalLightDiffuse;
	float3 TotalLightSpecular;

};

struct FDeferredLightingSplit
{
	float4 DiffuseLighting;
	float4 SpecularLighting;
};


void LightAccumulator_AddSplit(inout FLightAccumulator In, float3 DiffuseTotalLight, float3 SpecularTotalLight, float3 ScatterableLight, float3 CommonMultiplier, const bool bNeedsSeparateSubsurfaceLightAccumulation)
{

	In.TotalLight += (DiffuseTotalLight + SpecularTotalLight) * CommonMultiplier;


	if (bNeedsSeparateSubsurfaceLightAccumulation)
	{
		if ( 1  == 1)
		{
			if (View_bCheckerboardSubsurfaceProfileRendering == 0)
			{
				In.ScatterableLightLuma += Luminance(ScatterableLight * CommonMultiplier);
			}
		}
		else if ( 1  == 2)
		{

			In.ScatterableLight += ScatterableLight * CommonMultiplier;
		}
	}

	In.TotalLightDiffuse += DiffuseTotalLight * CommonMultiplier;
	In.TotalLightSpecular += SpecularTotalLight * CommonMultiplier;
}

void LightAccumulator_Add(inout FLightAccumulator In, float3 TotalLight, float3 ScatterableLight, float3 CommonMultiplier, const bool bNeedsSeparateSubsurfaceLightAccumulation)
{
	LightAccumulator_AddSplit(In, TotalLight, 0.0f, ScatterableLight, CommonMultiplier, bNeedsSeparateSubsurfaceLightAccumulation);
}




float4 LightAccumulator_GetResult(FLightAccumulator In)
{
	float4 Ret;

	if ( 0  == 1)
	{

		Ret = 0.1f * float4(1.0f, 0.25f, 0.075f, 0) * In.EstimatedCost;
	}
	else
	{
		Ret = float4(In.TotalLight, 0);

		if ( 1  == 1 )
		{
			if (View_bCheckerboardSubsurfaceProfileRendering == 0)
			{

				Ret.a = In.ScatterableLightLuma;
			}
		}
		else if ( 1  == 2)
		{


			Ret.a = Luminance(In.ScatterableLight);

		}
	}

	return Ret;
}


FDeferredLightingSplit LightAccumulator_GetResultSplit(FLightAccumulator In)
{
	float4 RetDiffuse;
	float4 RetSpecular;

	if ( 0  == 1)
	{

		RetDiffuse = 0.1f * float4(1.0f, 0.25f, 0.075f, 0) * In.EstimatedCost;
		RetSpecular = 0.1f * float4(1.0f, 0.25f, 0.075f, 0) * In.EstimatedCost;
	}
	else
	{
		RetDiffuse = float4(In.TotalLightDiffuse, 0);
		RetSpecular = float4(In.TotalLightSpecular, 0);

		if ( 1  == 1 )
		{
			if (View_bCheckerboardSubsurfaceProfileRendering == 0)
			{

				RetDiffuse.a = In.ScatterableLightLuma;
			}
		}
		else if ( 1  == 2)
		{


			RetDiffuse.a = Luminance(In.ScatterableLight);

		}
	}

	FDeferredLightingSplit Ret;
	Ret.DiffuseLighting = RetDiffuse;
	Ret.SpecularLighting = RetSpecular;

	return Ret;
}
#line 11 "/Engine/Private/DeferredShadingCommon.ush"
#line 12 "/Engine/Private/DeferredShadingCommon.ush"
#line 1 "MonteCarlo.ush"
#line 13 "/Engine/Private/MonteCarlo.ush"
float3x3 GetTangentBasis( float3 TangentZ )
{
	const float Sign = TangentZ.z >= 0 ? 1 : -1;
	const float a = -rcp( Sign + TangentZ.z );
	const float b = TangentZ.x * TangentZ.y * a;

	float3 TangentX = { 1 + Sign * a * Pow2( TangentZ.x ), Sign * b, -Sign * TangentZ.x };
	float3 TangentY = { b, Sign + a * Pow2( TangentZ.y ), -TangentZ.y };

	return float3x3( TangentX, TangentY, TangentZ );
}



float3x3 GetTangentBasisFrisvad(float3 TangentZ)
{
	float3 TangentX;
	float3 TangentY;

	if (TangentZ.z < -0.9999999f)
	{
		TangentX = float3(0, -1, 0);
		TangentY = float3(-1, 0, 0);
	}
	else
	{
		float A = 1.0f / (1.0f + TangentZ.z);
		float B = -TangentZ.x * TangentZ.y * A;
		TangentX = float3(1.0f - TangentZ.x * TangentZ.x * A, B, -TangentZ.x);
		TangentY = float3(B, 1.0f - TangentZ.y * TangentZ.y * A, -TangentZ.y);
	}

	return float3x3( TangentX, TangentY, TangentZ );
}

float3 TangentToWorld( float3 Vec, float3 TangentZ )
{
	return mul( Vec, GetTangentBasis( TangentZ ) );
}

float3 WorldToTangent(float3 Vec, float3 TangentZ)
{
	return mul(GetTangentBasis(TangentZ), Vec);
}

float2 Hammersley( uint Index, uint NumSamples, uint2 Random )
{
	float E1 = frac( (float)Index / NumSamples + float( Random.x & 0xffff ) / (1<<16) );
	float E2 = float( reversebits(Index) ^ Random.y ) * 2.3283064365386963e-10;
	return float2( E1, E2 );
}

float2 Hammersley16( uint Index, uint NumSamples, uint2 Random )
{
	float E1 = frac( (float)Index / NumSamples + float( Random.x ) * (1.0 / 65536.0) );
	float E2 = float( ( reversebits(Index) >> 16 ) ^ Random.y ) * (1.0 / 65536.0);
	return float2( E1, E2 );
}


float2 R2Sequence( uint Index )
{
	const float Phi = 1.324717957244746;
	const float2 a = float2( 1.0 / Phi, 1.0 / Pow2(Phi) );
	return frac( a * Index );
}



float2 JitteredR2( uint Index, uint NumSamples, float2 Jitter, float JitterAmount = 0.5 )
{
	const float Phi = 1.324717957244746;
	const float2 a = float2( 1.0 / Phi, 1.0 / Pow2(Phi) );
	const float d0 = 0.76;
	const float i0 = 0.7;

	return frac( a * float(Index) + ( JitterAmount * 0.5 * d0 * sqrt(PI) * rsqrt( float(NumSamples) ) ) * Jitter );
}


float2 JitteredR2( uint Index, float2 Jitter, float JitterAmount = 0.5 )
{
	const float Phi = 1.324717957244746;
	const float2 a = float2( 1.0 / Phi, 1.0 / Pow2(Phi) );
	const float d0 = 0.76;
	const float i0 = 0.7;

	return frac( a * Index + ( JitterAmount * 0.25 * d0 * sqrt(PI) * rsqrt( Index - i0 ) ) * Jitter );
}




float2 UniformSampleDisk( float2 E )
{
	float Theta = 2 * PI * E.x;
	float Radius = sqrt( E.y );
	return Radius * float2( cos( Theta ), sin( Theta ) );
}


float3 ConcentricDiskSamplingHelper(float2 E)
{
	float2 p = 2 * E - 1;
	float2 a = abs(p);
	float Lo = min(a.x, a.y);
	float Hi = max(a.x, a.y);
	float Epsilon = 5.42101086243e-20;
	float Phi = (PI / 4) * (Lo / (Hi + Epsilon) + 2 * float(a.y >= a.x));
	float Radius = Hi;

	const uint SignMask = 0x80000000;
	float2 Disk = asfloat((asuint(float2(cos(Phi), sin(Phi))) & ~SignMask) | (asuint(p) & SignMask));

	return float3(Disk, Radius);
}

float2 UniformSampleDiskConcentric( float2 E )
{
	float3 Result = ConcentricDiskSamplingHelper(E);
	return Result.xy * Result.z;
}



float2 UniformSampleDiskConcentricApprox( float2 E )
{
	float2 sf = E * sqrt(2.0) - sqrt(0.5);
	float2 sq = sf*sf;
	float root = sqrt(2.0*max(sq.x, sq.y) - min(sq.x, sq.y));
	if (sq.x > sq.y)
	{
		sf.x = sf.x > 0 ? root : -root;
	}
	else
	{
		sf.y = sf.y > 0 ? root : -root;
	}
	return sf;
}





float3 EquiAreaSphericalMapping(float2 UV)
{
	UV = 2 * UV - 1;
	float D = 1 - (abs(UV.x) + abs(UV.y));
	float R = 1 - abs(D);
	float Epsilon = 5.42101086243e-20;
	float Phi = (PI / 4) * ((abs(UV.y) - abs(UV.x)) / (R + Epsilon) + 1);
	float F = R * sqrt(2 - R * R);
	return float3(
		F * sign(UV.x) * abs(cos(Phi)),
		F * sign(UV.y) * abs(sin(Phi)),
		sign(D) * (1 - R * R)
	);
}




float2 InverseEquiAreaSphericalMapping(float3 Direction)
{
	float3 AbsDir = abs(Direction);
	float R = sqrt(1 - AbsDir.z);
	float Epsilon = 5.42101086243e-20;
	float x = min(AbsDir.x, AbsDir.y) / (max(AbsDir.x, AbsDir.y) + Epsilon);


	const float t1 = 0.406758566246788489601959989e-5f;
	const float t2 = 0.636226545274016134946890922156f;
	const float t3 = 0.61572017898280213493197203466e-2f;
	const float t4 = -0.247333733281268944196501420480f;
	const float t5 = 0.881770664775316294736387951347e-1f;
	const float t6 = 0.419038818029165735901852432784e-1f;
	const float t7 = -0.251390972343483509333252996350e-1f;


	float Phi = t6 + t7 * x;
	Phi = t5 + Phi * x;
	Phi = t4 + Phi * x;
	Phi = t3 + Phi * x;
	Phi = t2 + Phi * x;
	Phi = t1 + Phi * x;

	Phi = (AbsDir.x < AbsDir.y) ? 1 - Phi : Phi;
	float2 UV = float2(R - Phi * R, Phi * R);
	UV = (Direction.z < 0) ? 1 - UV.yx : UV;
	UV = asfloat(asuint(UV) ^ (asuint(Direction.xy) & 0x80000000u));
	return UV * 0.5 + 0.5;
}



float4 UniformSampleSphere( float2 E )
{
	float Phi = 2 * PI * E.x;
	float CosTheta = 1 - 2 * E.y;
	float SinTheta = sqrt( 1 - CosTheta * CosTheta );

	float3 H;
	H.x = SinTheta * cos( Phi );
	H.y = SinTheta * sin( Phi );
	H.z = CosTheta;

	float PDF = 1.0 / (4 * PI);

	return float4( H, PDF );
}


float4 UniformSampleHemisphere( float2 E )
{
	float Phi = 2 * PI * E.x;
	float CosTheta = E.y;
	float SinTheta = sqrt( 1 - CosTheta * CosTheta );

	float3 H;
	H.x = SinTheta * cos( Phi );
	H.y = SinTheta * sin( Phi );
	H.z = CosTheta;

	float PDF = 1.0 / (2 * PI);

	return float4( H, PDF );
}


float4 CosineSampleHemisphere( float2 E )
{
	float Phi = 2 * PI * E.x;
	float CosTheta = sqrt(E.y);
	float SinTheta = sqrt(1 - CosTheta * CosTheta);

	float3 H;
	H.x = SinTheta * cos(Phi);
	H.y = SinTheta * sin(Phi);
	H.z = CosTheta;

	float PDF = CosTheta * (1.0 / PI);

	return float4(H, PDF);
}


float4 CosineSampleHemisphereConcentric(float2 E)
{
	float3 Result = ConcentricDiskSamplingHelper(E);
	float SinTheta = Result.z;
	float CosTheta = sqrt(1 - SinTheta * SinTheta);
	return float4(Result.xy * SinTheta, CosTheta, CosTheta * (1.0 / PI));
}


float4 CosineSampleHemisphere( float2 E, float3 N )
{
	float3 H = UniformSampleSphere( E ).xyz;
	H = normalize( N + H );

	float PDF = dot(H, N) * (1.0 / PI);

	return float4( H, PDF );
}

float4 UniformSampleCone( float2 E, float CosThetaMax )
{
	float Phi = 2 * PI * E.x;
	float CosTheta = lerp( CosThetaMax, 1, E.y );
	float SinTheta = sqrt( 1 - CosTheta * CosTheta );

	float3 L;
	L.x = SinTheta * cos( Phi );
	L.y = SinTheta * sin( Phi );
	L.z = CosTheta;

	float PDF = 1.0 / ( 2 * PI * (1 - CosThetaMax) );

	return float4( L, PDF );
}




float4 UniformSampleConeRobust(float2 E, float SinThetaMax2)
{
	float Phi = 2 * PI * E.x;



	float OneMinusCosThetaMax = SinThetaMax2 < 0.01 ? SinThetaMax2 * (0.5 + 0.125 * SinThetaMax2) : 1 - sqrt(1 - SinThetaMax2);

	float CosTheta = 1 - OneMinusCosThetaMax * E.y;
	float SinTheta = sqrt(1 - CosTheta * CosTheta);

	float3 L;
	L.x = SinTheta * cos(Phi);
	L.y = SinTheta * sin(Phi);
	L.z = CosTheta;
	float PDF = 1.0 / (2 * PI * OneMinusCosThetaMax);

	return float4(L, PDF);
}

float UniformConeSolidAngle(float SinThetaMax2)
{
	float OneMinusCosThetaMax = SinThetaMax2 < 0.01 ? SinThetaMax2 * (0.5 + 0.125 * SinThetaMax2) : 1 - sqrt(1 - SinThetaMax2);
	return 2 * PI * OneMinusCosThetaMax;
}


float4 UniformSampleConeConcentricRobust(float2 E, float SinThetaMax2)
{



	float OneMinusCosThetaMax = SinThetaMax2 < 0.01 ? SinThetaMax2 * (0.5 + 0.125 * SinThetaMax2) : 1 - sqrt(1 - SinThetaMax2);
	float3 Result = ConcentricDiskSamplingHelper(E);
	float SinTheta = Result.z * sqrt(SinThetaMax2);
	float CosTheta = sqrt(1 - SinTheta * SinTheta);

	float3 L = float3(Result.xy * SinTheta, CosTheta);
	float PDF = 1.0 / (2 * PI * OneMinusCosThetaMax);

	return float4(L, PDF);
}


float4 ImportanceSampleGGX( float2 E, float a2 )
{
	float Phi = 2 * PI * E.x;
	float CosTheta = sqrt( (1 - E.y) / ( 1 + (a2 - 1) * E.y ) );
	float SinTheta = sqrt( 1 - CosTheta * CosTheta );

	float3 H;
	H.x = SinTheta * cos( Phi );
	H.y = SinTheta * sin( Phi );
	H.z = CosTheta;

	float d = ( CosTheta * a2 - CosTheta ) * CosTheta + 1;
	float D = a2 / ( PI*d*d );
	float PDF = D * CosTheta;

	return float4( H, PDF );
}

float VisibleGGXPDF(float3 V, float3 H, float a2)
{
	float NoV = V.z;
	float NoH = H.z;
	float VoH = dot(V, H);

	float d = (NoH * a2 - NoH) * NoH + 1;
	float D = a2 / (PI*d*d);

	float PDF = 2 * VoH * D / (NoV + sqrt(NoV * (NoV - NoV * a2) + a2));
	return PDF;
}

float VisibleGGXPDF_aniso(float3 V, float3 H, float2 Alpha)
{
	float NoV = V.z;
	float NoH = H.z;
	float VoH = dot(V, H);
	float a2 = Alpha.x * Alpha.y;
	float3 Hs = float3(Alpha.y * H.x, Alpha.x * H.y, a2 * NoH);
	float S = dot(Hs, Hs);
	float D = (1.0f / PI) * a2 * Square(a2 / S);
	float LenV = length(float3(V.x * Alpha.x, V.y * Alpha.y, NoV));
	float Pdf = (2 * D * VoH) / (NoV + LenV);
	return Pdf;
}





float4 ImportanceSampleVisibleGGX( float2 DiskE, float a2, float3 V )
{

	float a = sqrt(a2);


	float3 Vh = normalize( float3( a * V.xy, V.z ) );



	float LenSq = Vh.x * Vh.x + Vh.y * Vh.y;
	float3 Tangent0 = LenSq > 0 ? float3(-Vh.y, Vh.x, 0) * rsqrt(LenSq) : float3(1, 0, 0);
	float3 Tangent1 = cross(Vh, Tangent0);

	float2 p = DiskE;
	float s = 0.5 + 0.5 * Vh.z;
	p.y = (1 - s) * sqrt( 1 - p.x * p.x ) + s * p.y;

	float3 H;
	H = p.x * Tangent0;
	H += p.y * Tangent1;
	H += sqrt( saturate( 1 - dot( p, p ) ) ) * Vh;


	H = normalize( float3( a * H.xy, max(0.0, H.z) ) );

	return float4(H, VisibleGGXPDF(V, H, a2));
}





float4 ImportanceSampleVisibleGGX_aniso(float2 DiskE, float2 Alpha, float3 V)
{

	float3 Vh = normalize(float3(Alpha * V.xy, V.z));


	float LenSq = Vh.x * Vh.x + Vh.y * Vh.y;
	float3 Tx = LenSq > 0 ? float3(-Vh.y, Vh.x, 0) * rsqrt(LenSq) : float3(1, 0, 0);
	float3 Ty = cross(Vh, Tx);

	float2 p = DiskE;
	float s = 0.5 + 0.5 * Vh.z;
	p.y = lerp(sqrt(1 - p.x * p.x), p.y, s);

	float3 H = p.x * Tx + p.y * Ty + sqrt(saturate(1 - dot(p, p))) * Vh;


	H = normalize(float3(Alpha * H.xy, max(0.0, H.z)));

	return float4(H, VisibleGGXPDF_aniso(V, H, Alpha));
}



float MISWeight( uint Num, float PDF, uint OtherNum, float OtherPDF )
{
	float Weight = Num * PDF;
	float OtherWeight = OtherNum * OtherPDF;
	return Weight * Weight / (Weight * Weight + OtherWeight * OtherWeight);
}


float MISWeightRobust(float Pdf, float OtherPdf) {










	if (Pdf == OtherPdf)
	{

		return 0.5f;
	}






	if (OtherPdf < Pdf)
	{
		float x = OtherPdf / Pdf;
		return 1.0 / (1.0 + x * x);
	}
	else
	{

		float x = Pdf / OtherPdf;
		return 1.0 - 1.0 / (1.0 + x * x);
	}
}



float RayPDFToReflectionRayPDF(float VoH, float RayPDF)
{
	float ReflectPDF = RayPDF / (4.0 * saturate(VoH));

	return ReflectPDF;
}
#line 13 "/Engine/Private/DeferredShadingCommon.ush"
#line 1 "OctahedralCommon.ush"
#line 18 "/Engine/Private/OctahedralCommon.ush"
float2 UnitVectorToOctahedron( float3 N )
{
	N.xy /= dot( 1, abs(N) );
	if( N.z <= 0 )
	{
		N.xy = ( 1 - abs(N.yx) ) * ( N.xy >= 0 ? float2(1,1) : float2(-1,-1) );
	}
	return N.xy;
}

float3 OctahedronToUnitVector( float2 Oct )
{
	float3 N = float3( Oct, 1 - dot( 1, abs(Oct) ) );
	float t = max( -N.z, 0 );
	N.xy += N.xy >= 0 ? float2(-t, -t) : float2(t, t);
	return normalize(N);
}

float2 UnitVectorToHemiOctahedron( float3 N )
{
	N.xy /= dot( 1, abs(N) );
	return float2( N.x + N.y, N.x - N.y );
}

float3 HemiOctahedronToUnitVector( float2 Oct )
{
	Oct = float2( Oct.x + Oct.y, Oct.x - Oct.y );
	float3 N = float3( Oct, 2.0 - dot( 1, abs(Oct) ) );
	return normalize(N);
}


uint2 OctahedralMapWrapBorder(uint2 TexelCoord, uint Resolution, uint BorderSize)
{
	if (TexelCoord.x < BorderSize)
	{
		TexelCoord.x = BorderSize - 1 + BorderSize - TexelCoord.x;
		TexelCoord.y = Resolution - 1 - TexelCoord.y;
	}
	if (TexelCoord.x >= Resolution - BorderSize)
	{
		TexelCoord.x = (Resolution - BorderSize) - (TexelCoord.x - (Resolution - BorderSize - 1));
		TexelCoord.y = Resolution - 1 - TexelCoord.y;
	}
	if (TexelCoord.y < BorderSize)
	{
		TexelCoord.y = BorderSize - 1 + BorderSize - TexelCoord.y;
		TexelCoord.x = Resolution - 1 - TexelCoord.x;
	}
	if (TexelCoord.y >= Resolution - BorderSize)
	{
		TexelCoord.y = (Resolution - BorderSize) - (TexelCoord.y - (Resolution - BorderSize - 1));
		TexelCoord.x = Resolution - 1 - TexelCoord.x;
	}

	return TexelCoord - BorderSize;
}



float ComputeSphericalExcess(float3 A, float3 B, float3 C) {
    float CosAB = dot(A, B);
    float SinAB = 1.0f - CosAB * CosAB;
    float CosBC = dot(B, C);
    float SinBC = 1.0f - CosBC * CosBC;
    float CosCA = dot(C, A);
    float CosC = CosCA - CosAB * CosBC;
    float SinC = sqrt(SinAB * SinBC - CosC * CosC);
    float Inv = (1.0f - CosAB) * (1.0f - CosBC);
	return 2.0f * atan2(SinC, sqrt((SinAB * SinBC * (1.0f + CosBC) * (1.0f + CosAB)) / Inv) + CosC);
}


float OctahedralSolidAngle(float2 TexelCoord, float InvResolution)
{
	float3 Direction10 = OctahedronToUnitVector(TexelCoord + float2(.5f, -.5f) * InvResolution);
	float3 Direction01 = OctahedronToUnitVector(TexelCoord + float2(-.5f, .5f) * InvResolution);

	float SolidAngle0 = ComputeSphericalExcess(
		OctahedronToUnitVector(TexelCoord + float2(-.5f, -.5f) * InvResolution),
		Direction10,
		Direction01);

	float SolidAngle1 = ComputeSphericalExcess(
		OctahedronToUnitVector(TexelCoord + float2(.5f, .5f) * InvResolution),
		Direction01,
		Direction10);

	return SolidAngle0 + SolidAngle1;
}
#line 14 "/Engine/Private/DeferredShadingCommon.ush"
#line 27 "/Engine/Private/DeferredShadingCommon.ush"
uint bSceneLightingChannelsValid;


Texture2D SceneDepthTexture;
Texture2D<uint2> SceneStencilTexture;
Texture2D GBufferATexture;
Texture2D GBufferBTexture;
Texture2D GBufferCTexture;
Texture2D GBufferDTexture;
Texture2D GBufferETexture;
Texture2D GBufferVelocityTexture;
Texture2D GBufferFTexture;
Texture2D<uint> SceneLightingChannels;










float SampleDeviceZFromSceneTextures(float2 UV)
{
	return SceneDepthTexture.SampleLevel( D3DStaticPointClampedSampler , UV, 0).r;
}









float3 RGBToYCoCg( float3 RGB )
{
	float Y = dot( RGB, float3( 1, 2, 1 ) ) * 0.25;
	float Co = dot( RGB, float3( 2, 0, -2 ) ) * 0.25 + ( 0.5 * 256.0 / 255.0 );
	float Cg = dot( RGB, float3( -1, 2, -1 ) ) * 0.25 + ( 0.5 * 256.0 / 255.0 );

	float3 YCoCg = float3( Y, Co, Cg );
	return YCoCg;
}

float3 YCoCgToRGB( float3 YCoCg )
{
	float Y = YCoCg.x;
	float Co = YCoCg.y - ( 0.5 * 256.0 / 255.0 );
	float Cg = YCoCg.z - ( 0.5 * 256.0 / 255.0 );

	float R = Y + Co - Cg;
	float G = Y + Cg;
	float B = Y - Co - Cg;

	float3 RGB = float3( R, G, B );
	return RGB;
}

float3 Pack1212To888( float2 x )
{








	float2 x1212 = floor( x * 4095 );
	float2 High = floor( x1212 / 256 );
	float2 Low = x1212 - High * 256;
	float3 x888 = float3( Low, High.x + High.y * 16 );
	return saturate( x888 / 255 );

}

float2 Pack888To1212( float3 x )
{








	float3 x888 = floor( x * 255 );
	float High = floor( x888.z / 16 );
	float Low = x888.z - High * 16;
	float2 x1212 = x888.xy + float2( Low, High ) * 256;
	return saturate( x1212 / 4095 );

}

float3 EncodeNormal( float3 N )
{
	return N * 0.5 + 0.5;

}

float3 DecodeNormal( float3 N )
{
	return N * 2 - 1;

}

void EncodeNormal( inout float3 N, out uint Face )
{

	uint Axis = 2;
	if( abs(N.x) >= abs(N.y) && abs(N.x) >= abs(N.z) )
	{
		Axis = 0;
	}
	else if( abs(N.y) > abs(N.z) )
	{
		Axis = 1;
	}
	Face = Axis * 2;
#line 154 "/Engine/Private/DeferredShadingCommon.ush"
	N = Axis == 0 ? N.yzx : N;
	N = Axis == 1 ? N.xzy : N;

	float MaxAbs = 1.0 / sqrt(2.0);

	Face += N.z > 0 ? 0 : 1;
	N.xy *= N.z > 0 ? 1 : -1;
	N.xy = N.xy * (0.5 / MaxAbs) + 0.5;
}

void DecodeNormal( inout float3 N, in uint Face )
{
	uint Axis = Face >> 1;

	float MaxAbs = 1.0 / sqrt(2.0);

	N.xy = N.xy * (2 * MaxAbs) - (1 * MaxAbs);
	N.z = sqrt( 1 - dot( N.xy, N.xy ) );

	N = Axis == 0 ? N.zxy : N;
	N = Axis == 1 ? N.xzy : N;
	N *= (Face & 1) ? -1 : 1;
}

float3 EncodeBaseColor(float3 BaseColor)
{

	return BaseColor;
}

float3 DecodeBaseColor(float3 BaseColor)
{

	return BaseColor;
}

float3 EncodeSubsurfaceColor(float3 SubsurfaceColor)
{
	return sqrt(saturate(SubsurfaceColor));
}


float3 EncodeSubsurfaceProfile(float SubsurfaceProfile)
{
	return float3(SubsurfaceProfile, 0, 0);
}


float SubsurfaceDensityFromOpacity(float Opacity)
{
	return (-0.05f * log(1.0f - min(Opacity, 0.99f)));
}

float EncodeIndirectIrradiance(float IndirectIrradiance)
{
	float L = IndirectIrradiance;
	L *= View_PreExposure;
	const float LogBlackPoint = 0.00390625;
	return log2( L + LogBlackPoint ) / 16 + 0.5;
}

float DecodeIndirectIrradiance(float IndirectIrradiance)
{

	float LogL = IndirectIrradiance;
	const float LogBlackPoint = 0.00390625;
	return View_OneOverPreExposure * (exp2( LogL * 16 - 8 ) - LogBlackPoint);
}

float4 EncodeWorldTangentAndAnisotropy(float3 WorldTangent, float Anisotropy)
{
	return float4(
		EncodeNormal(WorldTangent),
		Anisotropy * 0.5f + 0.5f
		);
}

float ComputeAngleFromRoughness( float Roughness, const float Threshold = 0.04f )
{

	float Angle = 3 * Square( Roughness );
#line 240 "/Engine/Private/DeferredShadingCommon.ush"
	return Angle;
}

float ComputeRoughnessFromAngle( float Angle, const float Threshold = 0.04f )
{

	float Roughness = sqrt( 0.33333 * Angle );
#line 252 "/Engine/Private/DeferredShadingCommon.ush"
	return Roughness;
}

float AddAngleToRoughness( float Angle, float Roughness )
{
	return saturate( sqrt( Square( Roughness ) + 0.33333 * Angle ) );
}




float Encode71(float Scalar, uint Mask)
{
	return
		127.0f / 255.0f * saturate(Scalar) +
		128.0f / 255.0f * Mask;
}





float Decode71(float Scalar, out uint Mask)
{
	Mask = (uint)(Scalar > 0.5f);

	return (Scalar - 0.5f * Mask) * 2.0f;
}

float EncodeShadingModelIdAndSelectiveOutputMask(uint ShadingModelId, uint SelectiveOutputMask)
{
	uint Value = (ShadingModelId &  0xF ) | SelectiveOutputMask;
	return (float)Value / (float)0xFF;
}

uint DecodeShadingModelId(float InPackedChannel)
{
	return ((uint)round(InPackedChannel * (float)0xFF)) &  0xF ;
}

uint DecodeSelectiveOutputMask(float InPackedChannel)
{
	return ((uint)round(InPackedChannel * (float)0xFF)) & ~ 0xF ;
}

bool IsSubsurfaceModel(int ShadingModel)
{
	return ShadingModel ==  2
		|| ShadingModel ==  3
		|| ShadingModel ==  5
		|| ShadingModel ==  6
		|| ShadingModel ==  7
		|| ShadingModel ==  9 ;
}

bool UseSubsurfaceProfile(int ShadingModel)
{
	return ShadingModel ==  5  || ShadingModel ==  9 ;
}

bool HasCustomGBufferData(int ShadingModelID)
{
	return ShadingModelID ==  2
		|| ShadingModelID ==  3
		|| ShadingModelID ==  4
		|| ShadingModelID ==  5
		|| ShadingModelID ==  6
		|| ShadingModelID ==  7
		|| ShadingModelID ==  8
		|| ShadingModelID ==  9
		|| ShadingModelID ==  12 ;
}

bool HasAnisotropy(int SelectiveOutputMask)
{
	return (SelectiveOutputMask &  (1 << 4) ) != 0;
}


struct FGBufferData
{

	float3 WorldNormal;

	float3 WorldTangent;

	float3 DiffuseColor;

	float3 SpecularColor;

	float3 BaseColor;

	float Metallic;

	float Specular;

	float4 CustomData;

	float GenericAO;

	float IndirectIrradiance;


	float4 PrecomputedShadowFactors;

	float Roughness;

	float Anisotropy;

	float GBufferAO;

	uint DiffuseIndirectSampleOcclusion;

	uint ShadingModelID;

	uint SelectiveOutputMask;

	float PerObjectGBufferData;

	float CustomDepth;

	uint CustomStencil;


	float Depth;

	float4 Velocity;


	float3 StoredBaseColor;

	float StoredSpecular;

	float StoredMetallic;

	FHairMaterialParameterEx HairMaterialParameterEx;
};

bool CastContactShadow(FGBufferData GBufferData)
{
	uint PackedAlpha = (uint)(GBufferData.PerObjectGBufferData * 3.999f);
	bool bCastContactShadowBit = PackedAlpha & 1;

	bool bShadingModelCastContactShadows = (GBufferData.ShadingModelID !=  9 );
	return bCastContactShadowBit && bShadingModelCastContactShadows;
}

bool HasDynamicIndirectShadowCasterRepresentation(FGBufferData GBufferData)
{
	uint PackedAlpha = (uint)(GBufferData.PerObjectGBufferData * 3.999f);
	return (PackedAlpha & 2) != 0;
}




bool CheckerFromPixelPos(uint2 PixelPos)
{


	uint TemporalAASampleIndex = View_TemporalAAParams.x;


	return (PixelPos.x + PixelPos.y + TemporalAASampleIndex) % 2;
#line 419 "/Engine/Private/DeferredShadingCommon.ush"
}




bool CheckerFromSceneColorUV(float2 UVSceneColor)
{

	uint2 PixelPos = uint2(UVSceneColor * View_BufferSizeAndInvSize.xy);

	return CheckerFromPixelPos(PixelPos);
}
#line 433 "/Engine/Private/DeferredShadingCommon.ush"
#line 1 "GBufferHelpers.ush"
#line 10 "/Engine/Private/GBufferHelpers.ush"
float SquareInline(float X)
{
	return X * X;
}

float3 EncodeNormalHelper(float3 SrcNormal, float QuantizationBias)
{
	return SrcNormal * .5f + .5f;
}

float3 DecodeNormalHelper(float3 SrcNormal)
{
	return SrcNormal * 2.0f - 1.0f;
}


uint EncodeQuantize6(float Value, float QuantizationBias)
{
	return min(uint(saturate(Value) * 63.0f + .5f + QuantizationBias),63u);
}

float DecodeQuantize6(uint Value)
{
	return float(Value) / 63.0f;
}

uint EncodeQuantize6Sqrt(float Value, float QuantizationBias)
{
	return min(uint(sqrt(saturate(Value)) * 63.0f + .5f + QuantizationBias),63u);
}

float DecodeQuantize6Sqrt(uint Value)
{
	return SquareInline(float(Value) / 63.0f);
}

uint EncodeQuantize5(float Value, float QuantizationBias)
{
	return min(uint(saturate(Value) * 31.0f + .5f + QuantizationBias),31u);
}

float DecodeQuantize5(uint Value)
{
	return float(Value) / 31.0f;
}

uint EncodeQuantize5Sqrt(float Value, float QuantizationBias)
{
	return min(uint(sqrt(saturate(Value)) * 31.0f + .5f + QuantizationBias),31u);
}

float DecodeQuantize5Sqrt(uint Value)
{
	return SquareInline(float(Value) / 31.0f);
}

uint EncodeQuantize4(float Value, float QuantizationBias)
{
	return min(uint(saturate(Value) * 15.0f + .5f + QuantizationBias),15u);
}

float DecodeQuantize4(uint Value)
{
	return float(Value) / 15.0f;
}

uint EncodeQuantize4Sqrt(float Value, float QuantizationBias)
{
	return min(uint(sqrt(saturate(Value)) * 15.0f + .5f + QuantizationBias),15u);
}

float DecodeQuantize4Sqrt(uint Value)
{
	return SquareInline(float(Value) / 15.0f);
}


uint EncodeQuantize3(float Value, float QuantizationBias)
{
	return min(uint(saturate(Value) * 7.0f + .5f + QuantizationBias),7u);
}

float DecodeQuantize3(uint Value)
{
	return float(Value) / 7.0f;
}

uint EncodeQuantize3Sqrt(float Value, float QuantizationBias)
{
	return min(uint(sqrt(saturate(Value)) * 7.0f + .5f + QuantizationBias),7u);
}

float DecodeQuantize3Sqrt(uint Value)
{
	return SquareInline(float(Value) / 7.0f);
}

uint EncodeQuantize2(float Value, float QuantizationBias)
{
	return min(uint(saturate(Value) * 3.0f + .5f + QuantizationBias),3u);
}

float DecodeQuantize2(uint Value)
{
	return float(Value) / 3.0f;
}

uint EncodeQuantize2Sqrt(float Value, float QuantizationBias)
{
	return min(uint(sqrt(saturate(Value)) * 3.0f + .5f + QuantizationBias),3u);
}

float DecodeQuantize2Sqrt(uint Value)
{
	return SquareInline(float(Value) / 3.0f);
}

uint EncodeQuantize1(float Value, float QuantizationBias)
{
	return min(uint(saturate(Value) * 1.0f + .5f + QuantizationBias),1u);
}

float DecodeQuantize1(uint Value)
{
	return float(Value) / 1.0f;
}

uint EncodeQuantize1Sqrt(float Value, float QuantizationBias)
{
	return min(uint(sqrt(saturate(Value)) * 1.0f + .5f + QuantizationBias),1u);
}

float DecodeQuantize1Sqrt(uint Value)
{
	return SquareInline(float(Value) / 1.0f);
}


uint3 EncodeQuantize565(float3 Value, float QuantizationBias)
{
	uint3 Ret;
	Ret.x = EncodeQuantize5(Value.x,QuantizationBias);
	Ret.y = EncodeQuantize6(Value.y,QuantizationBias);
	Ret.z = EncodeQuantize5(Value.z,QuantizationBias);
	return Ret;
}

float3 DecodeQuantize565(uint3 Value)
{
	float3 Ret;
	Ret.x = DecodeQuantize5(Value.x);
	Ret.y = DecodeQuantize6(Value.y);
	Ret.z = DecodeQuantize5(Value.z);
	return Ret;
}

uint3 EncodeQuantize565Sqrt(float3 Value, float QuantizationBias)
{
	uint3 Ret;
	Ret.x = EncodeQuantize5Sqrt(Value.x,QuantizationBias);
	Ret.y = EncodeQuantize6Sqrt(Value.y,QuantizationBias);
	Ret.z = EncodeQuantize5Sqrt(Value.z,QuantizationBias);
	return Ret;
}

float3 DecodeQuantize565Sqrt(uint3 Value)
{
	float3 Ret;
	Ret.x = DecodeQuantize5Sqrt(Value.x);
	Ret.y = DecodeQuantize6Sqrt(Value.y);
	Ret.z = DecodeQuantize5Sqrt(Value.z);
	return Ret;
}


uint3 EncodeQuantize444(float3 Value, float QuantizationBias)
{
	uint3 Ret;
	Ret.x = EncodeQuantize4(Value.x,QuantizationBias);
	Ret.y = EncodeQuantize4(Value.y,QuantizationBias);
	Ret.z = EncodeQuantize4(Value.z,QuantizationBias);
	return Ret;
}

float3 DecodeQuantize444(uint3 Value)
{
	float3 Ret;
	Ret.x = DecodeQuantize4(Value.x);
	Ret.y = DecodeQuantize4(Value.y);
	Ret.z = DecodeQuantize4(Value.z);
	return Ret;
}

uint3 EncodeQuantize444Sqrt(float3 Value, float QuantizationBias)
{
	uint3 Ret;
	Ret.x = EncodeQuantize4Sqrt(Value.x,QuantizationBias);
	Ret.y = EncodeQuantize4Sqrt(Value.y,QuantizationBias);
	Ret.z = EncodeQuantize4Sqrt(Value.z,QuantizationBias);
	return Ret;
}

float3 DecodeQuantize444Sqrt(uint3 Value)
{
	float3 Ret;
	Ret.x = DecodeQuantize4Sqrt(Value.x);
	Ret.y = DecodeQuantize4Sqrt(Value.y);
	Ret.z = DecodeQuantize4Sqrt(Value.z);
	return Ret;
}


uint3 EncodeQuantize332(float3 Value, float QuantizationBias)
{
	uint3 Ret;
	Ret.x = EncodeQuantize3(Value.x,QuantizationBias);
	Ret.y = EncodeQuantize3(Value.y,QuantizationBias);
	Ret.z = EncodeQuantize2(Value.z,QuantizationBias);
	return Ret;
}

float3 DecodeQuantize332(uint3 Value)
{
	float3 Ret;
	Ret.x = DecodeQuantize3(Value.x);
	Ret.y = DecodeQuantize3(Value.y);
	Ret.z = DecodeQuantize2(Value.z);
	return Ret;
}

uint3 EncodeQuantize332Sqrt(float3 Value, float QuantizationBias)
{
	uint3 Ret;
	Ret.x = EncodeQuantize3Sqrt(Value.x,QuantizationBias);
	Ret.y = EncodeQuantize3Sqrt(Value.y,QuantizationBias);
	Ret.z = EncodeQuantize2Sqrt(Value.z,QuantizationBias);
	return Ret;
}

float3 DecodeQuantize332Sqrt(uint3 Value)
{
	float3 Ret;
	Ret.x = DecodeQuantize3Sqrt(Value.x);
	Ret.y = DecodeQuantize3Sqrt(Value.y);
	Ret.z = DecodeQuantize2Sqrt(Value.z);
	return Ret;
}


void EnvBRDFApproxFullyRoughHelper(inout float3 DiffuseColor, inout float3 SpecularColor)
{

	DiffuseColor += SpecularColor * 0.45;
	SpecularColor = 0;

}

void EnvBRDFApproxFullyRoughHelper(inout float3 DiffuseColor, inout float SpecularColor)
{
	DiffuseColor += SpecularColor * 0.45;
	SpecularColor = 0;
}








void GBufferPreEncode(inout FGBufferData GBuffer, bool bChecker, float GeometricAARoughness, inout  float3  OriginalBaseColor, inout  float  OriginalSpecular, inout  float  OriginalMetallic, float QuantizationBias)
{
#line 314 "/Engine/Private/GBufferHelpers.ush"
	GBuffer.DiffuseColor = OriginalBaseColor - OriginalBaseColor * OriginalMetallic;


	{

		GBuffer.DiffuseColor = GBuffer.DiffuseColor * View_DiffuseOverrideParameter.w + View_DiffuseOverrideParameter.xyz;
		GBuffer.SpecularColor = GBuffer.SpecularColor * View_SpecularOverrideParameter.w + View_SpecularOverrideParameter.xyz;
	}



	if (View_RenderingReflectionCaptureMask)

	{
		EnvBRDFApproxFullyRoughHelper(GBuffer.DiffuseColor, GBuffer.SpecularColor);

	}







		GBuffer.GenericAO = EncodeIndirectIrradiance(GBuffer.IndirectIrradiance * GBuffer.GBufferAO) + QuantizationBias * (1.0 / 255.0);
#line 345 "/Engine/Private/GBufferHelpers.ush"
}



void AdjustBaseColorAndSpecularColorForSubsurfaceProfileLightingCopyHack(inout float3 BaseColor, inout float3 SpecularColor, inout float Specular, bool bChecker)
{





	const bool bCheckerboardRequired = View_bSubsurfacePostprocessEnabled > 0 && View_bCheckerboardSubsurfaceProfileRendering > 0;
	BaseColor = View_bSubsurfacePostprocessEnabled ? float3(1, 1, 1) : BaseColor;

	if (bCheckerboardRequired)
	{



		BaseColor = bChecker;

		SpecularColor *= !bChecker;
		Specular *= !bChecker;
	}
}





void GBufferPostDecode(inout FGBufferData Ret, bool bChecker, bool bGetNormalizedNormal)
{
	Ret.CustomData = HasCustomGBufferData(Ret.ShadingModelID) ? Ret.CustomData : 0.0f;

	Ret.PrecomputedShadowFactors = !(Ret.SelectiveOutputMask & 0x2) ? Ret.PrecomputedShadowFactors : ((Ret.SelectiveOutputMask & 0x4) ? 0.0f : 1.0f);
	Ret.Velocity = !(Ret.SelectiveOutputMask & 0x8) ? Ret.Velocity : 0.0f;
	bool bHasAnisotropy = (Ret.SelectiveOutputMask & 0x1);

	Ret.StoredBaseColor = Ret.BaseColor;
	Ret.StoredMetallic = Ret.Metallic;
	Ret.StoredSpecular = Ret.Specular;






	Ret.GBufferAO = 1;
	Ret.DiffuseIndirectSampleOcclusion = 0x0;
	Ret.IndirectIrradiance = DecodeIndirectIrradiance(Ret.GenericAO.x);
#line 401 "/Engine/Private/GBufferHelpers.ush"
	if(bGetNormalizedNormal)
	{
		Ret.WorldNormal = normalize(Ret.WorldNormal);
	}

	[flatten]
	if( Ret.ShadingModelID ==  9  )
	{
		Ret.Metallic = 0.0;
#line 413 "/Engine/Private/GBufferHelpers.ush"
	}


	{
		Ret.SpecularColor = ComputeF0(Ret.Specular, Ret.BaseColor, Ret.Metallic);

		if (UseSubsurfaceProfile(Ret.ShadingModelID))
		{
			AdjustBaseColorAndSpecularColorForSubsurfaceProfileLightingCopyHack(Ret.BaseColor, Ret.SpecularColor, Ret.Specular, bChecker);
		}

		Ret.DiffuseColor = Ret.BaseColor - Ret.BaseColor * Ret.Metallic;


		{

			Ret.DiffuseColor = Ret.DiffuseColor * View_DiffuseOverrideParameter.www + View_DiffuseOverrideParameter.xyz;
			Ret.SpecularColor = Ret.SpecularColor * View_SpecularOverrideParameter.w + View_SpecularOverrideParameter.xyz;
		}

	}

	if (bHasAnisotropy)
	{
		Ret.WorldTangent = DecodeNormal(Ret.WorldTangent);
		Ret.Anisotropy = Ret.Anisotropy * 2.0f - 1.0f;

		if(bGetNormalizedNormal)
		{
			Ret.WorldTangent = normalize(Ret.WorldTangent);
		}
	}
	else
	{
		Ret.WorldTangent = 0;
		Ret.Anisotropy = 0;
	}



	Ret.SelectiveOutputMask = Ret.SelectiveOutputMask << 4;
}
#line 434 "/Engine/Private/DeferredShadingCommon.ush"
#line 435 "/Engine/Private/DeferredShadingCommon.ush"
#line 1 "/Engine/Generated/ShaderAutogen/AutogenShaderHeaders.ush"
#line 6 "/ShaderAutogen/PCD3D_SM5/AutogenShaderHeaders.ush"
float SampleDeviceZFromSceneTexturesTempCopy(float2 UV)
{
	return SceneDepthTexture.SampleLevel( D3DStaticPointClampedSampler , UV, 0).r;
}


void EncodeGBufferToMRT(inout FPixelShaderOut Out, FGBufferData GBuffer, float QuantizationBias)
{
	float4 MrtFloat1 = 0.0f;
	float4 MrtFloat2 = 0.0f;
	uint4 MrtUint2 = 0;
	float4 MrtFloat3 = 0.0f;
	float4 MrtFloat4 = 0.0f;
	float4 MrtFloat5 = 0.0f;

	float3 WorldNormal_Compressed = EncodeNormalHelper(GBuffer.WorldNormal, 0.0f);

	MrtFloat1.x = WorldNormal_Compressed.x;
	MrtFloat1.y = WorldNormal_Compressed.y;
	MrtFloat1.z = WorldNormal_Compressed.z;
	MrtFloat1.w = GBuffer.PerObjectGBufferData.x;
	MrtFloat2.x = GBuffer.Metallic.x;
	MrtFloat2.y = GBuffer.Specular.x;
	MrtFloat2.z = GBuffer.Roughness.x;
	MrtUint2.w |= ((((GBuffer.ShadingModelID.x) >> 0) & 0x0f) << 0);
	MrtUint2.w |= ((((GBuffer.SelectiveOutputMask.x) >> 0) & 0x0f) << 4);
	MrtFloat3.x = GBuffer.BaseColor.x;
	MrtFloat3.y = GBuffer.BaseColor.y;
	MrtFloat3.z = GBuffer.BaseColor.z;
	MrtFloat3.w = GBuffer.GenericAO.x;
	MrtFloat5.x = GBuffer.PrecomputedShadowFactors.x;
	MrtFloat5.y = GBuffer.PrecomputedShadowFactors.y;
	MrtFloat5.z = GBuffer.PrecomputedShadowFactors.z;
	MrtFloat5.w = GBuffer.PrecomputedShadowFactors.w;
	MrtFloat4.x = GBuffer.CustomData.x;
	MrtFloat4.y = GBuffer.CustomData.y;
	MrtFloat4.z = GBuffer.CustomData.z;
	MrtFloat4.w = GBuffer.CustomData.w;

	Out.MRT[1] = MrtFloat1;
	Out.MRT[2] = float4(MrtFloat2.x, MrtFloat2.y, MrtFloat2.z, (float(MrtUint2.w) + .5f) / 255.0f);
	Out.MRT[3] = MrtFloat3;
	Out.MRT[4] = MrtFloat4;
	Out.MRT[5] = MrtFloat5;
	Out.MRT[6] = float4(0.0f, 0.0f, 0.0f, 0.0f);
	Out.MRT[7] = float4(0.0f, 0.0f, 0.0f, 0.0f);
}


FGBufferData DecodeGBufferDataDirect(float4 InMRT1,
	float4 InMRT2,
	float4 InMRT3,
	float4 InMRT4,
	float4 InMRT5,

	float CustomNativeDepth,
	float4 AnisotropicData,
	uint CustomStencil,
	float SceneDepth,
	bool bGetNormalizedNormal,
	bool bChecker)
{
	FGBufferData Ret = (FGBufferData)0;
	float3 WorldNormal_Compressed = 0.0f;
	WorldNormal_Compressed.x = InMRT1.x;
	WorldNormal_Compressed.y = InMRT1.y;
	WorldNormal_Compressed.z = InMRT1.z;
	Ret.PerObjectGBufferData.x = InMRT1.w;
	Ret.Metallic.x = InMRT2.x;
	Ret.Specular.x = InMRT2.y;
	Ret.Roughness.x = InMRT2.z;
	Ret.ShadingModelID.x = (((uint((float(InMRT2.w) * 255.0f) + .5f) >> 0) & 0x0f) << 0);
	Ret.SelectiveOutputMask.x = (((uint((float(InMRT2.w) * 255.0f) + .5f) >> 4) & 0x0f) << 0);
	Ret.BaseColor.x = InMRT3.x;
	Ret.BaseColor.y = InMRT3.y;
	Ret.BaseColor.z = InMRT3.z;
	Ret.GenericAO.x = InMRT3.w;
	Ret.PrecomputedShadowFactors.x = InMRT5.x;
	Ret.PrecomputedShadowFactors.y = InMRT5.y;
	Ret.PrecomputedShadowFactors.z = InMRT5.z;
	Ret.PrecomputedShadowFactors.w = InMRT5.w;
	Ret.CustomData.x = InMRT4.x;
	Ret.CustomData.y = InMRT4.y;
	Ret.CustomData.z = InMRT4.z;
	Ret.CustomData.w = InMRT4.w;

	Ret.WorldNormal = DecodeNormalHelper(WorldNormal_Compressed);
	Ret.WorldTangent = AnisotropicData.xyz;
	Ret.Anisotropy = AnisotropicData.w;

	GBufferPostDecode(Ret,bChecker,bGetNormalizedNormal);

	Ret.CustomDepth = ConvertFromDeviceZ(CustomNativeDepth);
	Ret.CustomStencil = CustomStencil;
	Ret.Depth = SceneDepth;


	return Ret;
}





FGBufferData DecodeGBufferDataUV(float2 UV, bool bGetNormalizedNormal = true)
{
	float CustomNativeDepth = Texture2DSampleLevel(SceneTexturesStruct_CustomDepthTexture,  SceneTexturesStruct_PointClampSampler , UV, 0).r;
	int2 IntUV = (int2)trunc(UV * View_BufferSizeAndInvSize.xy);
	uint CustomStencil = SceneTexturesStruct_CustomStencilTexture.Load(int3(IntUV, 0))  .g ;
	float SceneDepth = CalcSceneDepth(UV);
	float4 AnisotropicData = Texture2DSampleLevel(SceneTexturesStruct_GBufferFTexture,  SceneTexturesStruct_PointClampSampler , UV, 0).xyzw;

	float4 InMRT1 = Texture2DSampleLevel(SceneTexturesStruct_GBufferATexture,  SceneTexturesStruct_PointClampSampler , UV, 0).xyzw;
	float4 InMRT2 = Texture2DSampleLevel(SceneTexturesStruct_GBufferBTexture,  SceneTexturesStruct_PointClampSampler , UV, 0).xyzw;
	float4 InMRT3 = Texture2DSampleLevel(SceneTexturesStruct_GBufferCTexture,  SceneTexturesStruct_PointClampSampler , UV, 0).xyzw;
	float4 InMRT4 = Texture2DSampleLevel(SceneTexturesStruct_GBufferDTexture,  SceneTexturesStruct_PointClampSampler , UV, 0).xyzw;
	float4 InMRT5 = Texture2DSampleLevel(SceneTexturesStruct_GBufferETexture,  SceneTexturesStruct_PointClampSampler , UV, 0).xyzw;

	FGBufferData Ret = DecodeGBufferDataDirect(InMRT1,
		InMRT2,
		InMRT3,
		InMRT4,
		InMRT5,

		CustomNativeDepth,
		AnisotropicData,
		CustomStencil,
		SceneDepth,
		bGetNormalizedNormal,
		CheckerFromSceneColorUV(UV));

	return Ret;
}



FGBufferData DecodeGBufferDataUint(uint2 PixelPos, bool bGetNormalizedNormal = true)
{
	float CustomNativeDepth = SceneTexturesStruct_CustomDepthTexture.Load(int3(PixelPos, 0)).r;
	uint CustomStencil = SceneTexturesStruct_CustomStencilTexture.Load(int3(PixelPos, 0))  .g ;
	float SceneDepth = CalcSceneDepth(PixelPos);
	float4 AnisotropicData = SceneTexturesStruct_GBufferFTexture.Load(int3(PixelPos, 0)).xyzw;

	float4 InMRT1 = SceneTexturesStruct_GBufferATexture.Load(int3(PixelPos, 0)).xyzw;
	float4 InMRT2 = SceneTexturesStruct_GBufferBTexture.Load(int3(PixelPos, 0)).xyzw;
	float4 InMRT3 = SceneTexturesStruct_GBufferCTexture.Load(int3(PixelPos, 0)).xyzw;
	float4 InMRT4 = SceneTexturesStruct_GBufferDTexture.Load(int3(PixelPos, 0)).xyzw;
	float4 InMRT5 = SceneTexturesStruct_GBufferETexture.Load(int3(PixelPos, 0)).xyzw;

	FGBufferData Ret = DecodeGBufferDataDirect(InMRT1,
		InMRT2,
		InMRT3,
		InMRT4,
		InMRT5,

		CustomNativeDepth,
		AnisotropicData,
		CustomStencil,
		SceneDepth,
		bGetNormalizedNormal,
		CheckerFromPixelPos(PixelPos));

	return Ret;
}



FGBufferData DecodeGBufferDataSceneTextures(float2 UV, bool bGetNormalizedNormal = true)
{
	uint CustomStencil = 0;
	float CustomNativeDepth = 0;
	float DeviceZ = SampleDeviceZFromSceneTexturesTempCopy(UV);
	float SceneDepth = ConvertFromDeviceZ(DeviceZ);
	float4 AnisotropicData = GBufferFTexture.SampleLevel( D3DStaticPointClampedSampler , UV, 0).xyzw;

	float4 InMRT1 = GBufferATexture.SampleLevel( D3DStaticPointClampedSampler , UV, 0).xyzw;
	float4 InMRT2 = GBufferBTexture.SampleLevel( D3DStaticPointClampedSampler , UV, 0).xyzw;
	float4 InMRT3 = GBufferCTexture.SampleLevel( D3DStaticPointClampedSampler , UV, 0).xyzw;
	float4 InMRT4 = GBufferDTexture.SampleLevel( D3DStaticPointClampedSampler , UV, 0).xyzw;
	float4 InMRT5 = GBufferETexture.SampleLevel( D3DStaticPointClampedSampler , UV, 0).xyzw;

	FGBufferData Ret = DecodeGBufferDataDirect(InMRT1,
		InMRT2,
		InMRT3,
		InMRT4,
		InMRT5,

		CustomNativeDepth,
		AnisotropicData,
		CustomStencil,
		SceneDepth,
		bGetNormalizedNormal,
		CheckerFromSceneColorUV(UV));

	return Ret;
}



FGBufferData DecodeGBufferDataSceneTexturesLoad(uint2 PixelCoord, bool bGetNormalizedNormal = true)
{
	uint CustomStencil = 0;
	float CustomNativeDepth = 0;
	float DeviceZ = SceneDepthTexture.Load(int3(PixelCoord, 0)).r;
	float SceneDepth = ConvertFromDeviceZ(DeviceZ);
	float4 AnisotropicData = GBufferFTexture.Load(int3(PixelCoord, 0)).xyzw;

	float4 InMRT1 = GBufferATexture.Load(int3(PixelCoord, 0)).xyzw;
	float4 InMRT2 = GBufferBTexture.Load(int3(PixelCoord, 0)).xyzw;
	float4 InMRT3 = GBufferCTexture.Load(int3(PixelCoord, 0)).xyzw;
	float4 InMRT4 = GBufferDTexture.Load(int3(PixelCoord, 0)).xyzw;
	float4 InMRT5 = GBufferETexture.Load(int3(PixelCoord, 0)).xyzw;

	FGBufferData Ret = DecodeGBufferDataDirect(InMRT1,
		InMRT2,
		InMRT3,
		InMRT4,
		InMRT5,

		CustomNativeDepth,
		AnisotropicData,
		CustomStencil,
		SceneDepth,
		bGetNormalizedNormal,
		CheckerFromPixelPos(PixelCoord));

	return Ret;
}
#line 436 "/Engine/Private/DeferredShadingCommon.ush"


struct FScreenSpaceData
{

	FGBufferData GBuffer;

	float AmbientOcclusion;
};


void SetGBufferForUnlit(out float4 OutGBufferB)
{
	OutGBufferB = 0;
	OutGBufferB.a = EncodeShadingModelIdAndSelectiveOutputMask( 0 , 0);
}



float4 ComputeIndirectLightingSampleE(uint2 TracingPixelCoord, uint TracingRayIndex, uint TracingRayCount)
{

	uint2 Seed0 = Rand3DPCG16(int3(TracingPixelCoord, View_StateFrameIndexMod8)).xy;
	uint2 Seed1 = Rand3DPCG16(int3(TracingPixelCoord + 17, View_StateFrameIndexMod8)).xy;

	return float4(
		Hammersley16(TracingRayIndex, TracingRayCount, Seed0),
		Hammersley16(TracingRayIndex, TracingRayCount, Seed1));
}



void EncodeGBuffer(
	FGBufferData GBuffer,
	out float4 OutGBufferA,
	out float4 OutGBufferB,
	out float4 OutGBufferC,
	out float4 OutGBufferD,
	out float4 OutGBufferE,
	out float4 OutGBufferVelocity,
	float QuantizationBias = 0
	)
{
	if (GBuffer.ShadingModelID ==  0 )
	{
		OutGBufferA = 0;
		SetGBufferForUnlit(OutGBufferB);
		OutGBufferC = 0;
		OutGBufferD = 0;
		OutGBufferE = 0;
	}
	else
	{





		OutGBufferA.rgb = EncodeNormal( GBuffer.WorldNormal );
		OutGBufferA.a = GBuffer.PerObjectGBufferData;
#line 506 "/Engine/Private/DeferredShadingCommon.ush"
		OutGBufferB.r = GBuffer.Metallic;
		OutGBufferB.g = GBuffer.Specular;
		OutGBufferB.b = GBuffer.Roughness;
		OutGBufferB.a = EncodeShadingModelIdAndSelectiveOutputMask(GBuffer.ShadingModelID, GBuffer.SelectiveOutputMask);

		OutGBufferC.rgb = EncodeBaseColor( GBuffer.BaseColor );





		OutGBufferC.a = EncodeIndirectIrradiance(GBuffer.IndirectIrradiance * GBuffer.GBufferAO) + QuantizationBias * (1.0 / 255.0);
#line 522 "/Engine/Private/DeferredShadingCommon.ush"
		OutGBufferD = GBuffer.CustomData;
		OutGBufferE = GBuffer.PrecomputedShadowFactors;
	}




	OutGBufferVelocity = 0;

}




bool AdjustBaseColorAndSpecularColorForSubsurfaceProfileLighting(inout float3 BaseColor, inout float Specular, bool bChecker)
{





	const bool bCheckerboardRequired = View_bSubsurfacePostprocessEnabled > 0 && View_bCheckerboardSubsurfaceProfileRendering > 0;
	BaseColor = View_bSubsurfacePostprocessEnabled ? float3(1, 1, 1) : BaseColor;

	if (bCheckerboardRequired)
	{


		BaseColor = bChecker;
		Specular *= !bChecker;
	}
	return bCheckerboardRequired;
}
void AdjustBaseColorAndSpecularColorForSubsurfaceProfileLighting(inout float3 BaseColor, inout float3 SpecularColor, inout float Specular, bool bChecker)
{
	const bool bCheckerboardRequired = AdjustBaseColorAndSpecularColorForSubsurfaceProfileLighting(BaseColor, Specular, bChecker);
	if (bCheckerboardRequired)
	{

		SpecularColor *= !bChecker;
	}
}



FGBufferData DecodeGBufferData(
	float4 InGBufferA,
	float4 InGBufferB,
	float4 InGBufferC,
	float4 InGBufferD,
	float4 InGBufferE,
	float4 InGBufferF,
	float4 InGBufferVelocity,
	float CustomNativeDepth,
	uint CustomStencil,
	float SceneDepth,
	bool bGetNormalizedNormal,
	bool bChecker)
{
	FGBufferData GBuffer;
	GBuffer.HairMaterialParameterEx = (FHairMaterialParameterEx)0;

	GBuffer.WorldNormal = DecodeNormal( InGBufferA.xyz );
	if(bGetNormalizedNormal)
	{
		GBuffer.WorldNormal = normalize(GBuffer.WorldNormal);
	}

	GBuffer.PerObjectGBufferData = InGBufferA.a;
	GBuffer.Metallic = InGBufferB.r;
	GBuffer.Specular = InGBufferB.g;
	GBuffer.Roughness = InGBufferB.b;



	GBuffer.ShadingModelID = DecodeShadingModelId(InGBufferB.a);
	GBuffer.SelectiveOutputMask = DecodeSelectiveOutputMask(InGBufferB.a);

	GBuffer.BaseColor = DecodeBaseColor(InGBufferC.rgb);






	GBuffer.GBufferAO = 1;
	GBuffer.DiffuseIndirectSampleOcclusion = 0x0;
	GBuffer.IndirectIrradiance = DecodeIndirectIrradiance(InGBufferC.a);
#line 616 "/Engine/Private/DeferredShadingCommon.ush"
	GBuffer.CustomData = HasCustomGBufferData(GBuffer.ShadingModelID) ? InGBufferD : 0;

	GBuffer.PrecomputedShadowFactors = !(GBuffer.SelectiveOutputMask &  (1 << 5) ) ? InGBufferE : ((GBuffer.SelectiveOutputMask &  (1 << 6) ) ? 0 : 1);
	GBuffer.CustomDepth = ConvertFromDeviceZ(CustomNativeDepth);
	GBuffer.CustomStencil = CustomStencil;
	GBuffer.Depth = SceneDepth;

	GBuffer.StoredBaseColor = GBuffer.BaseColor;
	GBuffer.StoredMetallic = GBuffer.Metallic;
	GBuffer.StoredSpecular = GBuffer.Specular;

	[flatten]
	if( GBuffer.ShadingModelID ==  9  )
	{
		GBuffer.Metallic = 0.0;
#line 634 "/Engine/Private/DeferredShadingCommon.ush"
	}


	{
		GBuffer.SpecularColor = ComputeF0(GBuffer.Specular, GBuffer.BaseColor, GBuffer.Metallic);

		if (UseSubsurfaceProfile(GBuffer.ShadingModelID))
		{
			AdjustBaseColorAndSpecularColorForSubsurfaceProfileLighting(GBuffer.BaseColor, GBuffer.SpecularColor, GBuffer.Specular, bChecker);
		}

		GBuffer.DiffuseColor = GBuffer.BaseColor - GBuffer.BaseColor * GBuffer.Metallic;


		{

			GBuffer.DiffuseColor = GBuffer.DiffuseColor * View_DiffuseOverrideParameter.www + View_DiffuseOverrideParameter.xyz;
			GBuffer.SpecularColor = GBuffer.SpecularColor * View_SpecularOverrideParameter.w + View_SpecularOverrideParameter.xyz;
		}

	}

	{
		bool bHasAnisoProp = HasAnisotropy(GBuffer.SelectiveOutputMask);

		GBuffer.WorldTangent = bHasAnisoProp ? DecodeNormal(InGBufferF.rgb) : 0;
		GBuffer.Anisotropy = bHasAnisoProp ? InGBufferF.a * 2.0f - 1.0f : 0;

		if (bGetNormalizedNormal && bHasAnisoProp)
		{
			GBuffer.WorldTangent = normalize(GBuffer.WorldTangent);
		}
	}

	GBuffer.Velocity = !(GBuffer.SelectiveOutputMask &  (1 << 7) ) ? InGBufferVelocity : 0;

	return GBuffer;
}

float3 ExtractSubsurfaceColor(FGBufferData BufferData)
{
	return Square(BufferData.CustomData.rgb);
}

uint ExtractSubsurfaceProfileInt(float ProfileNormFloat)
{
	return uint(ProfileNormFloat * 255.0f + 0.5f);
}

uint ExtractSubsurfaceProfileInt(FGBufferData BufferData)
{
	return ExtractSubsurfaceProfileInt(BufferData.CustomData.r);
}





	FGBufferData GetGBufferDataUint(uint2 PixelPos, bool bGetNormalizedNormal = true)
	{

		return DecodeGBufferDataUint(PixelPos,bGetNormalizedNormal);
#line 722 "/Engine/Private/DeferredShadingCommon.ush"
	}


	FScreenSpaceData GetScreenSpaceDataUint(uint2 PixelPos, bool bGetNormalizedNormal = true)
	{
		FScreenSpaceData Out;

		Out.GBuffer = GetGBufferDataUint(PixelPos, bGetNormalizedNormal);

		float4 ScreenSpaceAO = Texture2DSampleLevel(SceneTexturesStruct_ScreenSpaceAOTexture,  SceneTexturesStruct_PointClampSampler , (PixelPos + 0.5f) * View_BufferSizeAndInvSize.zw, 0);
		Out.AmbientOcclusion = ScreenSpaceAO.r;

		return Out;
	}







FGBufferData GetGBufferDataFromSceneTextures(float2 UV, bool bGetNormalizedNormal = true)
{

	return DecodeGBufferDataSceneTextures(UV,bGetNormalizedNormal);
#line 765 "/Engine/Private/DeferredShadingCommon.ush"
}


uint GetSceneLightingChannel(uint2 PixelCoord)
{
	[branch]
	if (bSceneLightingChannelsValid)
	{
		return SceneLightingChannels.Load(uint3(PixelCoord, 0)).x;
	}
	return ~0;
}




FGBufferData GetGBufferData(float2 UV, bool bGetNormalizedNormal = true)
{

	return DecodeGBufferDataUV(UV,bGetNormalizedNormal);
#line 813 "/Engine/Private/DeferredShadingCommon.ush"
}


uint GetShadingModelId(float2 UV)
{
	return DecodeShadingModelId(Texture2DSampleLevel(SceneTexturesStruct_GBufferBTexture,  SceneTexturesStruct_PointClampSampler , UV, 0).a);
}


FScreenSpaceData GetScreenSpaceData(float2 UV, bool bGetNormalizedNormal = true)
{
	FScreenSpaceData Out;

	Out.GBuffer = GetGBufferData(UV, bGetNormalizedNormal);
	float4 ScreenSpaceAO = Texture2DSampleLevel(SceneTexturesStruct_ScreenSpaceAOTexture,  SceneTexturesStruct_PointClampSampler , UV, 0);

	Out.AmbientOcclusion = ScreenSpaceAO.r;

	return Out;
}



float3 AOMultiBounce(float3 BaseColor, float AO)
{
	float3 a = 2.0404 * BaseColor - 0.3324;
	float3 b = -4.7951 * BaseColor + 0.6417;
	float3 c = 2.7552 * BaseColor + 0.6903;
	return max(AO, ((AO * a + b) * AO + c) * AO);
}
#line 7 "/Engine/Private/SceneTextureParameters.ush"
#line 4 "/Plugin/NRD/Private/REBLUR_PackInputData.cs.usf"
#line 6 "/Plugin/NRD/Private/REBLUR_PackInputData.cs.usf"
#line 1 "NRD.ush"
#line 192 "/Plugin/NRD/Private/NRD.ush"
float3 _NRD_DecodeUnitVector( float2 p, const bool bSigned = false, const bool bNormalize = true )
{
    p = bSigned ? p : ( p * 2.0 - 1.0 );


    float3 n = float3( p.xy, 1.0 - abs( p.x ) - abs( p.y ) );
    float t = saturate( -n.z );
    n.xy += n.xy >= 0.0 ? -t : t;

    return bNormalize ? normalize( n ) : n;
}

float2 _NRD_EncodeUnitVector(float3 n, const bool bSigned = false)
{
	n = normalize(n);

	n /= (abs(n.x) + abs(n.y) + abs(n.z));
	if (n.z < 0.0)
	{
		n.xy = (1.0 - abs(n.yx)) * (n.xy >= 0.0 ? 1.0 : -1.0);
	}
	float2 p = n.xy;

	p = bSigned ? p : (p * 0.5 + 0.5);

	return p;
}

float _NRD_Luminance( float3 linearColor )
{
    return dot( linearColor, float3( 0.2990, 0.5870, 0.1140 ) );
}

float _NRD_GetColorCompressionExposureForSpatialPasses( float linearRoughness )
{
#line 241 "/Plugin/NRD/Private/NRD.ush"
        return 0.5 * ( 1.0 - linearRoughness ) / ( 1.0 + 1000.0 * linearRoughness * linearRoughness ) + ( 1.0 - sqrt( saturate( linearRoughness ) ) ) * 0.03;
#line 249 "/Plugin/NRD/Private/NRD.ush"
}


float _REBLUR_GetHitDistanceNormalization( float viewZ, float4 hitDistParams, float meterToUnitsMultiplier, float linearRoughness = 1.0 )
{
    return ( hitDistParams.x * meterToUnitsMultiplier + abs( viewZ ) * hitDistParams.y ) * lerp( 1.0, hitDistParams.z, saturate( exp2( hitDistParams.w * linearRoughness * linearRoughness ) ) );
}










float4 NRD_FrontEnd_UnpackNormalAndRoughness( float4 p, out float materialID )
{
    float4 r;






        r.xyz = p.xyz * 2.0 - 1.0;
        r.w = p.w;






        materialID = 0;



    r.xyz = normalize( r.xyz );
#line 293 "/Plugin/NRD/Private/NRD.ush"
    return r;
}

float4 NRD_FrontEnd_PackNormalAndRoughness(float3 n, float r, float materialID)
{
	float4 p;
#line 309 "/Plugin/NRD/Private/NRD.ush"
	p.xyz = n.xyz * 2.0 - 1.0;
	p.w = r;









	return p;
}


float4 NRD_FrontEnd_PackDirectionAndPdf( float3 direction, float pdf )
{
    return float4( direction, pdf );
}

float4 NRD_FrontEnd_UnpackDirectionAndPdf( float4 directionAndPdf )
{
    directionAndPdf.w = max( directionAndPdf.w,  0.01  );

    return directionAndPdf;
}






float REBLUR_FrontEnd_GetNormHitDist( float hitDist, float viewZ, float4 hitDistParams, float meterToUnitsMultiplier, float linearRoughness = 1.0 )
{
    float f = _REBLUR_GetHitDistanceNormalization( viewZ, hitDistParams, meterToUnitsMultiplier, linearRoughness );

    return saturate( hitDist / f );
}

float4 REBLUR_FrontEnd_PackRadianceAndHitDist( float3 radiance, float normHitDist, bool sanitize = true )
{
    if( sanitize )
    {
        radiance = any( isnan( radiance ) | isinf( radiance ) ) ? 0 : clamp( radiance, 0,  65504.0  );
        normHitDist = ( isnan( normHitDist ) | isinf( normHitDist ) ) ? 0 : saturate( normHitDist );
    }

    return float4( radiance, normHitDist );
}





float4 RELAX_FrontEnd_PackRadianceAndHitDist( float3 radiance, float hitDist, bool sanitize = true )
{
    if( sanitize )
    {
        radiance = any( isnan( radiance ) | isinf( radiance ) ) ? 0 : clamp( radiance, 0,  65504.0  );
        hitDist = ( isnan( hitDist ) | isinf( hitDist ) ) ? 0 : clamp( hitDist, 0,  65504.0  );
    }

    return float4( radiance, hitDist );
}
#line 384 "/Plugin/NRD/Private/NRD.ush"
float2 SIGMA_FrontEnd_PackShadow( float viewZ, float distanceToOccluder, float tanOfLightAngularRadius )
{
    float2 r;
    r.x = 0.0;
    r.y = clamp( viewZ *  0.0125 , - 65504.0 ,  65504.0  );

    [flatten]
    if( distanceToOccluder ==  65504.0  )
        r.x =  65504.0 ;
    else if( distanceToOccluder != 0.0 )
    {
        float distanceToOccluderProj = distanceToOccluder * tanOfLightAngularRadius;
        r.x = clamp( distanceToOccluderProj,  0.0001 , 32768.0 );
    }

    return r;
}

float2 SIGMA_FrontEnd_PackShadow( float viewZ, float distanceToOccluder, float tanOfLightAngularRadius, float3 translucency, out float4 shadowTranslucency )
{
    shadowTranslucency.x = float( distanceToOccluder ==  65504.0  );
    shadowTranslucency.yzw = saturate( translucency );

    return SIGMA_FrontEnd_PackShadow( viewZ, distanceToOccluder, tanOfLightAngularRadius );
}





float2x3  SIGMA_FrontEnd_MultiLightStart()
{
    return (  float2x3  )0;
}

void SIGMA_FrontEnd_MultiLightUpdate( float3 L, float distanceToOccluder, float tanOfLightAngularRadius, float weight, inout  float2x3  multiLightShadowData )
{
    float shadow = float( distanceToOccluder ==  65504.0  );
    float distanceToOccluderProj = SIGMA_FrontEnd_PackShadow( 0, distanceToOccluder, tanOfLightAngularRadius ).x;


    multiLightShadowData[ 0 ] += L * shadow;


    weight *= _NRD_Luminance( L );

    multiLightShadowData[ 1 ] += float3( distanceToOccluderProj * weight, weight, 0 );
}

float2 SIGMA_FrontEnd_MultiLightEnd( float viewZ,  float2x3  multiLightShadowData, float3 Lsum, out float4 shadowTranslucency )
{
    shadowTranslucency.yzw = multiLightShadowData[ 0 ] / max( Lsum, 1e-6 );
    shadowTranslucency.x = _NRD_Luminance( shadowTranslucency.yzw );

    float2 r;
    r.x = multiLightShadowData[ 1 ].x / max( multiLightShadowData[ 1 ].y, 1e-6 );
    r.y = clamp( viewZ *  0.0125 , - 65504.0 ,  65504.0  );

    return r;
}
#line 481 "/Plugin/NRD/Private/NRD.ush"
float NRD_GetCorrectedHitDist( float hitDist, float bounceIndex, float roughness0 = 1.0, float importance = 1.0 )
{


    float fade = lerp( 1.0, bounceIndex, roughness0 );

    return hitDist * importance / ( fade * fade );
}
#line 499 "/Plugin/NRD/Private/NRD.ush"
float NRD_GetTrimmingFactor( float roughness, float3 trimmingParams )
{
    float trimmingFactor = trimmingParams.x * smoothstep( trimmingParams.y, trimmingParams.z, roughness );

    return trimmingFactor;
}


float NRD_GetSampleWeight( float3 radiance, bool sanitize = true )
{
    return ( any( isnan( radiance ) | isinf( radiance ) ) && sanitize ) ? 0.0 : 1.0;
}
#line 7 "/Plugin/NRD/Private/REBLUR_PackInputData.cs.usf"

RWTexture2D<float4> OutNormalAndRoughness;
RWTexture2D<float> OutViewDepth;
RWTexture2D<float> OutDiffHitDist;
RWTexture2D<float3> OutMotionVector;

Texture2D<float> InAmbientOcclusionMask;

[numthreads(8, 8, 1)]
void main(uint2 DispatchThreadId : SV_DispatchThreadId)
{
    uint2 PixelCoord = DispatchThreadId;
	uint2 extent;

	OutNormalAndRoughness.GetDimensions(extent.x, extent.y);

	if(any(PixelCoord >= View_ViewSizeAndInvSize.xy))
	{
		OutNormalAndRoughness[PixelCoord] = 1.f;
		return;
	}

	float4 GBufferA = GBufferATexture[PixelCoord];
	float4 GBufferB = GBufferBTexture[PixelCoord];
	float DeviceZ = SceneDepthTexture[PixelCoord].x;
	float Depth = ConvertFromDeviceZ(DeviceZ);

	OutNormalAndRoughness[PixelCoord] = float4(normalize(DecodeNormal(GBufferA.xyz)) * 0.5 + 0.5, GBufferB.b);
	OutViewDepth[PixelCoord] = Depth;
	OutDiffHitDist[PixelCoord] = InAmbientOcclusionMask[PixelCoord];

	float4 EncodedVelocity = GBufferVelocityTexture[PixelCoord];
	float3 Velocity;
	if (all(EncodedVelocity.xy > 0))
	{
		Velocity = DecodeVelocityFromTexture(EncodedVelocity);
	}
	else
	{
		float4 ClipPos;
		ClipPos.xy = SvPositionToScreenPosition(float4(PixelCoord.xy, 0, 1)).xy;
		ClipPos.z = DeviceZ;
		ClipPos.w = 1;

		float4 PrevClipPos = mul(ClipPos, View_ClipToPrevClip);

		if (PrevClipPos.w > 0)
		{
			PrevClipPos /= PrevClipPos.w;
			Velocity = ClipPos.xyz - PrevClipPos.xyz;
		}
		else
		{
			Velocity = EncodedVelocity;
		}
	}

	float3 OutVelocity = float3(Velocity.xy * float2(0.5, -0.5), Velocity.z);
	OutMotionVector[PixelCoord] = -OutVelocity.xyz;
}
