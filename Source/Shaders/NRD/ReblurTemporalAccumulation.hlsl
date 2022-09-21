#line 1 "ParseAndMoveShaderParametersToRootConstantBuffer"
cbuffer _RootShaderParameters
{
float4x4 gViewToClip : packoffset(c0);
float4x4 gViewToWorld : packoffset(c4);
float4 gFrustum : packoffset(c8);
float4 gHitDistParams : packoffset(c9);
float4 gViewVectorWorld : packoffset(c10);
float4 gViewVectorWorldPrev : packoffset(c11);
float2 gInvScreenSize : packoffset(c12);
float2 gScreenSize : packoffset(c12.z);
float2 gInvRectSize : packoffset(c13);
float2 gRectSize : packoffset(c13.z);
float2 gRectSizePrev : packoffset(c14);
float2 gResolutionScale : packoffset(c14.z);
float2 gRectOffset : packoffset(c15);
float2 gSensitivityToDarkness : packoffset(c15.z);
uint2 gRectOrigin : packoffset(c16);
float gReference : packoffset(c16.z);
float gOrthoMode : packoffset(c16.w);
float gUnproject : packoffset(c17);
float gDebug : packoffset(c17.y);
float gDenoisingRange : packoffset(c17.z);
float gPlaneDistSensitivity : packoffset(c17.w);
float gFramerateScale : packoffset(c18);
float gBlurRadius : packoffset(c18.y);
float gMaxAccumulatedFrameNum : packoffset(c18.z);
float gAntiFirefly : packoffset(c18.w);
float gMinConvergedStateBaseRadiusScale : packoffset(c19.y);
float gLobeAngleFraction : packoffset(c19.z);
float gRoughnessFraction : packoffset(c19.w);
float gResponsiveAccumulationRoughnessThreshold : packoffset(c20);
float gDiffPrepassBlurRadius : packoffset(c20.y);
float gSpecPrepassBlurRadius : packoffset(c20.z);
uint gIsWorldSpaceMotionEnabled : packoffset(c20.w);
uint gFrameIndex : packoffset(c21);
uint gResetHistory : packoffset(c21.y);
uint gDiffMaterialMask : packoffset(c21.z);
uint gSpecMaterialMask : packoffset(c21.w);
float4x4 gWorldToViewPrev : packoffset(c22);
float4x4 gWorldToClipPrev : packoffset(c26);
float4x4 gWorldToClip : packoffset(c30);
float4x4 gWorldPrevToWorld : packoffset(c34);
float4 gFrustumPrev : packoffset(c38);
float4 gCameraDelta : packoffset(c39);
float4 gRotator : packoffset(c40);
float2 gMotionVectorScale : packoffset(c41);
float gCheckerboardResolveAccumSpeed : packoffset(c41.z);
float gDisocclusionThreshold : packoffset(c41.w);
uint gDiffCheckerboard : packoffset(c42);
uint gSpecCheckerboard : packoffset(c42.y);
uint gIsPrepassEnabled : packoffset(c42.z);
}

#line 1 "/Plugin/NRD/Private/Reblur/REBLUR_DiffuseOcclusion_TemporalAccumulation.cs.usf"
#line 11 "/Plugin/NRD/Private/Reblur/REBLUR_DiffuseOcclusion_TemporalAccumulation.cs.usf"
#line 1 "BindingBridge.ush"
#line 19 "/Plugin/NRD/Private/Reblur/BindingBridge.ush"
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
#line 20 "/Plugin/NRD/Private/Reblur/BindingBridge.ush"
#line 12 "/Plugin/NRD/Private/Reblur/REBLUR_DiffuseOcclusion_TemporalAccumulation.cs.usf"
#line 13 "/Plugin/NRD/Private/Reblur/REBLUR_DiffuseOcclusion_TemporalAccumulation.cs.usf"
#line 1 "STL.ush"
#line 32 "/Plugin/NRD/Private/Reblur/STL.ush"
namespace STL
{
    static const float FLT_MIN = 1e-15;





    namespace Math
    {



        float Pi( float x )
        { return  radians( 180.0 * x ) ; }

        float2 Pi( float2 x )
        { return  radians( 180.0 * x ) ; }

        float3 Pi( float3 x )
        { return  radians( 180.0 * x ) ; }

        float4 Pi( float4 x )
        { return  radians( 180.0 * x ) ; }




        float RadToDeg( float x )
        { return  ( x * 180.0 / Pi( 1.0 ) ) ; }

        float2 RadToDeg( float2 x )
        { return  ( x * 180.0 / Pi( 1.0 ) ) ; }

        float3 RadToDeg( float3 x )
        { return  ( x * 180.0 / Pi( 1.0 ) ) ; }

        float4 RadToDeg( float4 x )
        { return  ( x * 180.0 / Pi( 1.0 ) ) ; }




        float DegToRad( float x )
        { return  ( x * Pi( 1.0 ) / 180.0 ) ; }

        float2 DegToRad( float2 x )
        { return  ( x * Pi( 1.0 ) / 180.0 ) ; }

        float3 DegToRad( float3 x )
        { return  ( x * Pi( 1.0 ) / 180.0 ) ; }

        float4 DegToRad( float4 x )
        { return  ( x * Pi( 1.0 ) / 180.0 ) ; }




        void Swap( inout uint x, inout uint y )
        {  x ^= y; y ^= x; x ^= y ; }

        void Swap( inout uint2 x, inout uint2 y )
        {  x ^= y; y ^= x; x ^= y ; }

        void Swap( inout uint3 x, inout uint3 y )
        {  x ^= y; y ^= x; x ^= y ; }

        void Swap( inout uint4 x, inout uint4 y )
        {  x ^= y; y ^= x; x ^= y ; }

        void Swap( inout float x, inout float y )
        { float t = x; x = y; y = t; }

        void Swap( inout float2 x, inout float2 y )
        { float2 t = x; x = y; y = t; }

        void Swap( inout float3 x, inout float3 y )
        { float3 t = x; x = y; y = t; }

        void Swap( inout float4 x, inout float4 y )
        { float4 t = x; x = y; y = t; }





        float LinearStep( float a, float b, float x )
        { return  saturate( ( x - a ) / ( b - a ) ) ; }

        float2 LinearStep( float2 a, float2 b, float2 x )
        { return  saturate( ( x - a ) / ( b - a ) ) ; }

        float3 LinearStep( float3 a, float3 b, float3 x )
        { return  saturate( ( x - a ) / ( b - a ) ) ; }

        float4 LinearStep( float4 a, float4 b, float4 x )
        { return  saturate( ( x - a ) / ( b - a ) ) ; }





        float SmoothStep01( float x )
        { return  ( saturate( x ) * saturate( x ) * ( 3.0 - 2.0 * saturate( x ) ) ) ; }

        float2 SmoothStep01( float2 x )
        { return  ( saturate( x ) * saturate( x ) * ( 3.0 - 2.0 * saturate( x ) ) ) ; }

        float3 SmoothStep01( float3 x )
        { return  ( saturate( x ) * saturate( x ) * ( 3.0 - 2.0 * saturate( x ) ) ) ; }

        float4 SmoothStep01( float4 x )
        { return  ( saturate( x ) * saturate( x ) * ( 3.0 - 2.0 * saturate( x ) ) ) ; }

        float SmoothStep( float a, float b, float x )
        { x =  saturate( ( x - a ) / ( b - a ) ) ; return  ( x * x * ( 3.0 - 2.0 * x ) ) ; }

        float2 SmoothStep( float2 a, float2 b, float2 x )
        { x =  saturate( ( x - a ) / ( b - a ) ) ; return  ( x * x * ( 3.0 - 2.0 * x ) ) ; }

        float3 SmoothStep( float3 a, float3 b, float3 x )
        { x =  saturate( ( x - a ) / ( b - a ) ) ; return  ( x * x * ( 3.0 - 2.0 * x ) ) ; }

        float4 SmoothStep( float4 a, float4 b, float4 x )
        { x =  saturate( ( x - a ) / ( b - a ) ) ; return  ( x * x * ( 3.0 - 2.0 * x ) ) ; }






        float SmootherStep( float a, float b, float x )
        { x =  saturate( ( x - a ) / ( b - a ) ) ; return  ( x * x * x * ( x * ( x * 6.0 - 15.0 ) + 10.0 ) ) ; }

        float2 SmootherStep( float2 a, float2 b, float2 x )
        { x =  saturate( ( x - a ) / ( b - a ) ) ; return  ( x * x * x * ( x * ( x * 6.0 - 15.0 ) + 10.0 ) ) ; }

        float3 SmootherStep( float3 a, float3 b, float3 x )
        { x =  saturate( ( x - a ) / ( b - a ) ) ; return  ( x * x * x * ( x * ( x * 6.0 - 15.0 ) + 10.0 ) ) ; }

        float4 SmootherStep( float4 a, float4 b, float4 x )
        { x =  saturate( ( x - a ) / ( b - a ) ) ; return  ( x * x * x * ( x * ( x * 6.0 - 15.0 ) + 10.0 ) ) ; }






        float Sign( float x,   const uint mode =  1  )
        { return mode ==  1  ?  ( step( 0.0, x ) * 2.0 - 1.0 )  : sign( x ); }

        float2 Sign( float2 x,   const uint mode =  1  )
        { return mode ==  1  ?  ( step( 0.0, x ) * 2.0 - 1.0 )  : sign( x ); }

        float3 Sign( float3 x,   const uint mode =  1  )
        { return mode ==  1  ?  ( step( 0.0, x ) * 2.0 - 1.0 )  : sign( x ); }

        float4 Sign( float4 x,   const uint mode =  1  )
        { return mode ==  1  ?  ( step( 0.0, x ) * 2.0 - 1.0 )  : sign( x ); }


        float Pow( float x, float y )
        { return pow( abs( x ), y ); }

        float2 Pow( float2 x, float y )
        { return pow( abs( x ), y ); }

        float2 Pow( float2 x, float2 y )
        { return pow( abs( x ), y ); }

        float3 Pow( float3 x, float y )
        { return pow( abs( x ), y ); }

        float3 Pow( float3 x, float3 y )
        { return pow( abs( x ), y ); }

        float4 Pow( float4 x, float y )
        { return pow( abs( x ), y ); }

        float4 Pow( float4 x, float4 y )
        { return pow( abs( x ), y ); }


        float Pow01( float x, float y )
        { return pow( saturate( x ), y ); }

        float2 Pow01( float2 x, float y )
        { return pow( saturate( x ), y ); }

        float2 Pow01( float2 x, float2 y )
        { return pow( saturate( x ), y ); }

        float3 Pow01( float3 x, float y )
        { return pow( saturate( x ), y ); }

        float3 Pow01( float3 x, float3 y )
        { return pow( saturate( x ), y ); }

        float4 Pow01( float4 x, float y )
        { return pow( saturate( x ), y ); }

        float4 Pow01( float4 x, float4 y )
        { return pow( saturate( x ), y ); }





        float Sqrt( float x,   const uint mode =  1  )
        { return sqrt( mode ==  1  ? max( x, 0 ) : x ); }

        float2 Sqrt( float2 x,   const uint mode =  1  )
        { return sqrt( mode ==  1  ? max( x, 0 ) : x ); }

        float3 Sqrt( float3 x,   const uint mode =  1  )
        { return sqrt( mode ==  1  ? max( x, 0 ) : x ); }

        float4 Sqrt( float4 x,   const uint mode =  1  )
        { return sqrt( mode ==  1  ? max( x, 0 ) : x ); }


        float Sqrt01( float x )
        { return sqrt( saturate( x ) ); }

        float2 Sqrt01( float2 x )
        { return sqrt( saturate( x ) ); }

        float3 Sqrt01( float3 x )
        { return sqrt( saturate( x ) ); }

        float4 Sqrt01( float4 x )
        { return sqrt( saturate( x ) ); }







        float Rsqrt( float x,   const uint mode =  3  )
        {
            if( mode <=  1  )
                return rsqrt( mode ==  0  ? x : max( x, FLT_MIN ) );

            return 1.0 / sqrt( mode ==  2  ? x : max( x, FLT_MIN ) );
        }

        float2 Rsqrt( float2 x,   const uint mode =  3  )
        {
            if( mode <=  1  )
                return rsqrt( mode ==  0  ? x : max( x, FLT_MIN ) );

            return 1.0 / sqrt( mode ==  2  ? x : max( x, FLT_MIN ) );
        }

        float3 Rsqrt( float3 x,   const uint mode =  3  )
        {
            if( mode <=  1  )
                return rsqrt( mode ==  0  ? x : max( x, FLT_MIN ) );

            return 1.0 / sqrt( mode ==  2  ? x : max( x, FLT_MIN ) );
        }

        float4 Rsqrt( float4 x,   const uint mode =  3  )
        {
            if( mode <=  1  )
                return rsqrt( mode ==  0  ? x : max( x, FLT_MIN ) );

            return 1.0 / sqrt( mode ==  2  ? x : max( x, FLT_MIN ) );
        }





        float AcosApprox( float x )
        { return  ( sqrt( 2.0 ) * sqrt( saturate( 1.0 - x ) ) ) ; }

        float2 AcosApprox( float2 x )
        { return  ( sqrt( 2.0 ) * sqrt( saturate( 1.0 - x ) ) ) ; }

        float3 AcosApprox( float3 x )
        { return  ( sqrt( 2.0 ) * sqrt( saturate( 1.0 - x ) ) ) ; }

        float4 AcosApprox( float4 x )
        { return  ( sqrt( 2.0 ) * sqrt( saturate( 1.0 - x ) ) ) ; }







        float PositiveRcp( float x,   const uint mode =  3  )
        {
            if( mode <=  1  )
                return rcp( mode ==  0  ? x : max( x, FLT_MIN ) );

            return 1.0 / ( mode ==  2  ? x : max( x, FLT_MIN ) );
        }

        float2 PositiveRcp( float2 x,   const uint mode =  3  )
        {
            if( mode <=  1  )
                return rcp( mode ==  0  ? x : max( x, FLT_MIN ) );

            return 1.0 / ( mode ==  2  ? x : max( x, FLT_MIN ) );
        }

        float3 PositiveRcp( float3 x,   const uint mode =  3  )
        {
            if( mode <=  1  )
                return rcp( mode ==  0  ? x : max( x, FLT_MIN ) );

            return 1.0 / ( mode ==  2  ? x : max( x, FLT_MIN ) );
        }

        float4 PositiveRcp( float4 x,   const uint mode =  3  )
        {
            if( mode <=  1  )
                return rcp( mode ==  0  ? x : max( x, FLT_MIN ) );

            return 1.0 / ( mode ==  2  ? x : max( x, FLT_MIN ) );
        }


        float LengthSquared( float2 v )
        { return dot( v, v ); }

        float LengthSquared( float3 v )
        { return dot( v, v ); }

        float LengthSquared( float4 v )
        { return dot( v, v ); }


        uint ReverseBits4( uint x )
        {
            x = ( ( x & 0x5 ) << 1 ) | ( ( x & 0xA ) >> 1 );
            x = ( ( x & 0x3 ) << 2 ) | ( ( x & 0xC ) >> 2 );

            return x;
        }

        uint ReverseBits8( uint x )
        {
            x = ( ( x & 0x55 ) << 1 ) | ( ( x & 0xAA ) >> 1 );
            x = ( ( x & 0x33 ) << 2 ) | ( ( x & 0xCC ) >> 2 );
            x = ( ( x & 0x0F ) << 4 ) | ( ( x & 0xF0 ) >> 4 );

            return x;
        }

        uint ReverseBits16( uint x )
        {
            x = ( ( x & 0x5555 ) << 1 ) | ( ( x & 0xAAAA ) >> 1 );
            x = ( ( x & 0x3333 ) << 2 ) | ( ( x & 0xCCCC ) >> 2 );
            x = ( ( x & 0x0F0F ) << 4 ) | ( ( x & 0xF0F0 ) >> 4 );
            x = ( ( x & 0x00FF ) << 8 ) | ( ( x & 0xFF00 ) >> 8 );

            return x;
        }

        uint ReverseBits32( uint x )
        {
            x = ( x << 16 ) | ( x >> 16 );
            x = ( ( x & 0x55555555 ) << 1 ) | ( ( x & 0xAAAAAAAA ) >> 1 );
            x = ( ( x & 0x33333333 ) << 2 ) | ( ( x & 0xCCCCCCCC ) >> 2 );
            x = ( ( x & 0x0F0F0F0F ) << 4 ) | ( ( x & 0xF0F0F0F0 ) >> 4 );
            x = ( ( x & 0x00FF00FF ) << 8 ) | ( ( x & 0xFF00FF00 ) >> 8 );

            return x;
        }

        uint CompactBits( uint x )
        {
            x &= 0x55555555;
            x = ( x ^ ( x >> 1 ) ) & 0x33333333;
            x = ( x ^ ( x >> 2 ) ) & 0x0F0F0F0F;
            x = ( x ^ ( x >> 4 ) ) & 0x00FF00FF;
            x = ( x ^ ( x >> 8 ) ) & 0x0000FFFF;

            return x;
        }
    };





    namespace Geometry
    {
        float2 GetPerpendicular( float2 v )
        {
            return float2( -v.y, v.x );
        }

        float4 GetRotator( float angle )
        {
            float ca = cos( angle );
            float sa = sin( angle );

            return float4( ca, sa, -sa, ca );
        }

        float4 GetRotator( float sa, float ca )
        { return float4( ca, sa, -sa, ca ); }

        float3x3 GetRotator( float3 axis, float angle )
        {
            float sa = sin( angle );
            float ca = cos( angle );
            float one_ca = 1.0 - ca;

            float3 a = sa * axis;
            float3 b = one_ca * axis.xyx * axis.yzz;

            float3 t1 = one_ca * ( axis * axis ) + ca;
            float3 t2 = b.xyz - a.zxy;
            float3 t3 = b.zxy + a.yzx;

            return float3x3
            (
                t1.x, t2.x, t3.x,
                t3.y, t1.y, t2.y,
                t2.z, t3.z, t1.z
            );
        }

        float4 CombineRotators( float4 r1, float4 r2 )
        { return r1.xyxy * r2.xxzz + r1.zwzw * r2.yyww; }

        float2 RotateVector( float4 rotator, float2 v )
        { return v.x * rotator.xz + v.y * rotator.yw; }

        float3 RotateVector( float4x4 m, float3 v )
        { return mul( ( float3x3 )m, v ); }

        float3 RotateVector( float3x3 m, float3 v )
        { return mul( m, v ); }

        float2 RotateVectorInverse( float4 rotator, float2 v )
        { return v.x * rotator.xy + v.y * rotator.zw; }

        float3 RotateVectorInverse( float4x4 m, float3 v )
        { return mul( ( float3x3 )transpose( m ), v ); }

        float3 RotateVectorInverse( float3x3 m, float3 v )
        { return mul( transpose( m ), v ); }

        float3 AffineTransform( float4x4 m, float3 p )
        { return mul( m, float4( p, 1.0 ) ).xyz; }

        float3 AffineTransform( float3x4 m, float3 p )
        { return mul( m, float4( p, 1.0 ) ); }

        float3 AffineTransform( float4x4 m, float4 p )
        { return mul( m, p ).xyz; }

        float4 ProjectiveTransform( float4x4 m, float3 p )
        { return mul( m, float4( p, 1.0 ) ); }

        float4 ProjectiveTransform( float4x4 m, float4 p )
        { return mul( m, p ); }

        float3 GetPerpendicularVector( float3 N )
        {
            float3 T = float3( N.z, -N.x, N.y );
            T -= N * dot( T, N );

            return normalize( T );
        }


        float3x3 GetBasis( float3 N )
        {
            float sz = Math::Sign( N.z );
            float a = 1.0 / ( sz + N.z );
            float ya = N.y * a;
            float b = N.x * ya;
            float c = N.x * sz;

            float3 T = float3( c * N.x * a - 1.0, sz * b, c );
            float3 B = float3( b, N.y * ya - sz, N.y );



            return float3x3( T, B, N );
        }

        float2 GetBarycentricCoords( float3 p, float3 a, float3 b, float3 c )
        {
            float3 v0 = b - a;
            float3 v1 = c - a;
            float3 v2 = p - a;

            float d00 = dot( v0, v0 );
            float d01 = dot( v0, v1 );
            float d11 = dot( v1, v1 );
            float d20 = dot( v2, v0 );
            float d21 = dot( v2, v1 );

            float2 barys;
            barys.x = d11 * d20 - d01 * d21;
            barys.y = d00 * d21 - d01 * d20;

            float invDenom = 1.0 / ( d00 * d11 - d01 * d01 );

            return barys * invDenom;
        }

        float DistanceAttenuation( float dist, float Rmax )
        {

            float falloff = dist / Rmax;
            falloff *= falloff;
            falloff = saturate( 1.0 - falloff * falloff );
            falloff *= falloff;

            float atten = falloff;
            atten *= Math::PositiveRcp( dist * dist + 1.0 );

            return atten;
        }

        float3 UnpackLocalNormal( float2 localNormal, bool isUnorm = true )
        {
            float3 n;
            n.xy = isUnorm ? ( localNormal * ( 255.0 / 127.0 ) - 1.0 ) : localNormal;
            n.z = Math::Sqrt01( 1.0 - Math::LengthSquared( n.xy ) );

            return n;
        }

        float3 TransformLocalNormal( float2 localNormal, float4 T, float3 N )
        {
            float3 n = UnpackLocalNormal( localNormal );
            float3 B = cross( N, T.xyz );

            return normalize( T.xyz * n.x + B * n.y * T.w + N * n.z );
        }

        float SolidAngle( float cosHalfAngle )
        {
            return Math::Pi( 2.0 ) * ( 1.0 - cosHalfAngle );
        }

        float3 ReconstructViewPosition( float2 uv, float4 cameraFrustum, float viewZ = 1.0, float isOrtho = 0.0 )
        {
            float3 p;
            p.xy = uv * cameraFrustum.zw + cameraFrustum.xy;
            p.xy *= viewZ * ( 1.0 - abs( isOrtho ) ) + isOrtho;
            p.z = viewZ;

            return p;
        }

        float2 GetScreenUv( float4x4 worldToClip, float3 X )
        {
            float4 clip = Geometry::ProjectiveTransform( worldToClip, X );
            float2 uv = ( clip.xy / clip.w ) * float2( 0.5, -0.5 ) + 0.5;
            uv = clip.w < 0.0 ? 99999.0 : uv;

            return uv;
        }




        float2 GetPrevUvFromMotion( float2 uv, float3 X, float4x4 worldToClipPrev, float3 motionVector,   const uint motionType =  1  )
        {
            float3 Xprev = X + motionVector;
            float2 uvPrev = GetScreenUv( worldToClipPrev, Xprev );

            [flatten]
            if( motionType ==  0  )
                uvPrev = uv + motionVector.xy;

            return uvPrev;
        }
    };





    namespace Color
    {





        float Luminance( float3 linearColor,   const uint mode =  0  )
        {
            return dot( linearColor, mode ==  0  ? float3( 0.2990, 0.5870, 0.1140 ) : float3( 0.2126, 0.7152, 0.0722 ) );
        }

        float3 Saturation( float3 color, float amount )
        {
            float luma = Luminance( color );

            return lerp( color, luma, amount );
        }


        float3 LinearToGamma( float3 color, float gamma = 2.2 )
        {
            return Math::Pow01( color, 1.0 / gamma );
        }

        float3 GammaToLinear( float3 color, float gamma = 2.2 )
        {
            return Math::Pow01( color, gamma );
        }

        float3 LinearToSrgb( float3 color )
        {
            const float4 consts = float4( 1.055, 0.41666, -0.055, 12.92 );
            color = saturate( color );

            return lerp( consts.x * Math::Pow( color, consts.yyy ) + consts.zzz, consts.w * color, color < 0.0031308 );
        }

        float3 SrgbToLinear( float3 color )
        {
            const float4 consts = float4( 1.0 / 12.92, 1.0 / 1.055, 0.055 / 1.055, 2.4 );
            color = saturate( color );

            return lerp( color * consts.x, Math::Pow( color * consts.y + consts.zzz, consts.www ), color > 0.04045 );
        }










        float3 LinearToPq( float3 color )
        {
            float3 L = color /  10000.0 ;
            float3 Lm = Math::Pow( L,  0.1593017578125  );
            float3 N = (  0.8359375  +  18.8515625  * Lm ) * Math::PositiveRcp( 1.0 +  18.6875  * Lm );

            return Math::Pow( N,  78.84375  );
        }

        float3 PqToLinear( float3 color )
        {
            float3 Np = Math::Pow( color, 1.0 /  78.84375  );
            float3 L = Np -  0.8359375 ;
            L *= Math::PositiveRcp(  18.8515625  -  18.6875  * Np );
            L = Math::Pow( L, 1.0 /  0.1593017578125  );

            return L *  10000.0 ;
        }

        float3 LinearToYCoCg( float3 color )
        {
            float Co = color.x - color.z;
            float t = color.z + Co * 0.5;
            float Cg = color.y - t;
            float Y = t + Cg * 0.5;


            Y = max( Y, 0.0 );

            return float3( Y, Co, Cg );
        }

        float3 YCoCgToLinear( float3 color )
        {

            color.x = max( color.x, 0.0 );

            float t = color.x - color.z * 0.5;
            float g = color.z + t;
            float b = t - color.y * 0.5;
            float r = b + color.y;
            float3 res = float3( r, g, b );

            return res;
        }


        float3 GammaToXyz( float3 color )
        {
            static const float3x3 M =
            {
                0.4123907992659595, 0.3575843393838780, 0.1804807884018343,
                0.2126390058715104, 0.7151686787677559, 0.0721923153607337,
                0.0193308187155918, 0.1191947797946259, 0.9505321522496608
            };

            return mul( M, color );
        }

        float3 XyzToGamma( float3 color )
        {
            static const float3x3 M =
            {
                3.240969941904522, -1.537383177570094, -0.4986107602930032,
                -0.9692436362808803, 1.875967501507721, 0.04155505740717569,
                0.05563007969699373, -0.2039769588889765, 1.056971514242878
            };

            return mul( M, color );
        }




        uint LinearToLogLuv( float3 color )
        {

            float3 XYZ = GammaToXyz( color );



            float logY = 409.6 * ( log2( XYZ.y ) + 20.0 );
            uint Le = uint( clamp( logY, 0.0, 16383.0 ) );



            if( Le == 0 )
                return 0;









            float invDenom = 1.0 / ( -2.0 * XYZ.x + 12.0 * XYZ.y + 3.0 * ( XYZ.x + XYZ.y + XYZ.z ) );
            float2 uv = float2( 4.0, 9.0 ) * XYZ.xy * invDenom;



            uint2 uve = uint2( clamp( 820.0 * uv, 0.0, 511.0 ) );

            return ( Le << 18 ) | ( uve.x << 9 ) | uve.y;
        }


        float3 LogLuvToLinear( uint packedColor )
        {

            uint Le = packedColor >> 18;
            if( Le == 0 )
                return 0;

            float logY = ( float( Le ) + 0.5 ) / 409.6 - 20.0;
            float Y = exp2( logY );




            uint2 uve = uint2( packedColor >> 9, packedColor ) & 0x1ff;
            float2 uv = ( float2( uve ) + 0.5 ) / 820.0;

            float invDenom = 1.0 / ( 6.0 * uv.x - 16.0 * uv.y + 12.0 );
            float2 xy = float2( 9.0, 4.0 ) * uv * invDenom;




            float s = Y / xy.y;
            float3 XYZ = float3( s * xy.x, Y, s * ( 1.0 - xy.x - xy.y ) );


            float3 color = max( XyzToGamma( XYZ ), 0.0 );

            return color;
        }


        float3 Compress( float3 color, float exposure = 1.0 )
        {
            float luma = Luminance( color );

            return color * Math::PositiveRcp( 1.0 + luma * exposure );
        }

        float3 Decompress( float3 color, float exposure = 1.0 )
        {
            float luma = Luminance( color );

            return color * Math::PositiveRcp( 1.0 - luma * exposure );
        }

        float3 HdrToLinear( float3 colorMulExposure )
        {
            float3 x0 = colorMulExposure * 0.38317;
            float3 x1 = GammaToLinear( 1.0 - exp( -colorMulExposure ) );
            float3 color = lerp( x0, x1, step( 1.413, colorMulExposure ) );

            return saturate( color );
        }

        float3 LinearToHdr( float3 color )
        {
            float3 x0 = color / 0.38317;
            float3 x1 = -log( max( 1.0 - LinearToGamma( color ), 1e-6 ) );
            float3 colorMulExposure = lerp( x0, x1, step( 1.413, x0 ) );

            return colorMulExposure;
        }

        float3 HdrToGamma( float3 colorMulExposure )
        {
            float3 x0 = LinearToGamma( colorMulExposure * 0.38317 );
            float3 x1 = 1.0 - exp( -colorMulExposure );

            x0 = lerp( x0, x1, step( 1.413, colorMulExposure ) );

            return saturate( x0 );
        }

        float3 _UnchartedCurve( float3 color )
        {
            float A = 0.22;
            float B = 0.3;
            float C = 0.1;
            float D = 0.2;
            float E = 0.01;
            float F = 0.3;

            return saturate( ( ( color * ( A * color + C * B ) + D * E ) / ( color * ( A * color + B ) + D * F ) ) - ( E / F ) );
        }

        float3 HdrToLinear_Uncharted( float3 color )
        {

            return saturate( _UnchartedCurve( color ) / _UnchartedCurve( 11.2 ).x );
        }

        float3 HdrToLinear_Aces( float3 color )
        {

            color *= 0.6;

            float A = 2.51;
            float B = 0.03;
            float C = 2.43;
            float D = 0.59;
            float E = 0.14;

            return saturate( ( color * ( A * color + B ) ) * Math::PositiveRcp( color * ( C * color + D ) + E ) );
        }


        float4 BlendSoft( float4 a, float4 b )
        {
            float4 t = 1.0 - 2.0 * b;
            float4 c = ( 2.0 * b + a * t ) * a;
            float4 d = 2.0 * a * ( 1.0 - b ) - Math::Sqrt( a ) * t;
            bool4 res = b > 0.5;

            return lerp( c, d, res );
        }

        float4 BlendDarken( float4 a, float4 b )
        {
            bool4 res = a > b;

            return lerp( a, b, res );
        }

        float4 BlendDifference( float4 a, float4 b )
        {
            return abs( a - b );
        }

        float4 BlendScreen( float4 a, float4 b )
        {
            return a + b * ( 1.0 - a );
        }

        float4 BlendOverlay( float4 a, float4 b )
        {
            bool4 res = a > 0.5;
            float4 c = 2.0 * a * b;
            float4 d = 2.0 * BlendScreen( a, b ) - 1.0;

            return lerp( c, d, res );
        }





        float4 Clamp( float4 m1, float4 sigma, float4 prevSample,   const uint mode =  1  )
        {
            float4 a = m1 - sigma;
            float4 b = m1 + sigma;
            float4 clampedSample = clamp( prevSample, a, b );

            if( mode ==  1  )
            {

                float3 d = prevSample.xyz - m1.xyz;
                float3 dn = abs( d * Math::PositiveRcp( sigma.xyz ) );
                float maxd = max( dn.x, max( dn.y, dn.z ) );
                float3 t = m1.xyz + d * Math::PositiveRcp( maxd );

                clampedSample.xyz = maxd > 1.0 ? t : prevSample.xyz;
            }

            return clampedSample;
        }

        float Clamp( float m1, float sigma, float prevSample )
        {
            float a = m1 - sigma;
            float b = m1 + sigma;

            return clamp( prevSample, a, b );
        }


        float3 ColorizeLinear( float x )
        {
            x = saturate( x );

            float3 color;
            if( x < 0.25 )
                color = lerp( float3( 0, 0, 0 ), float3( 0, 0, 1 ), Math::SmoothStep( 0.00, 0.25, x ) );
            else if( x < 0.50 )
                color = lerp( float3( 0, 0, 1 ), float3( 0, 1, 0 ), Math::SmoothStep( 0.25, 0.50, x ) );
            else if( x < 0.75 )
                color = lerp( float3( 0, 1, 0 ), float3( 1, 1, 0 ), Math::SmoothStep( 0.50, 0.75, x ) );
            else
                color = lerp( float3( 1, 1, 0 ), float3( 1, 0, 0 ), Math::SmoothStep( 0.75, 1.00, x ) );

            return color;
        }


        float3 ColorizeZucconi( float x )
        {

            x = saturate( x ) * 0.85;

            const float3 c1 = float3( 3.54585104, 2.93225262, 2.41593945 );
            const float3 x1 = float3( 0.69549072, 0.49228336, 0.27699880 );
            const float3 y1 = float3( 0.02312639, 0.15225084, 0.52607955 );

            float3 t = c1 * ( x - x1 );
            float3 a = saturate( 1.0 - t * t - y1 );

            const float3 c2 = float3( 3.90307140, 3.21182957, 3.96587128 );
            const float3 x2 = float3( 0.11748627, 0.86755042, 0.66077860 );
            const float3 y2 = float3( 0.84897130, 0.88445281, 0.73949448 );

            float3 k = c2 * ( x - x2 );
            float3 b = saturate( 1.0 - k * k - y2 );

            return saturate( a + b );
        }
    };





    namespace Packing
    {



        uint RgbaToUint( float4 c,   const uint Rbits,   const uint Gbits = 0,   const uint Bbits = 0,   const uint Abits = 0 )
        {
            const uint Rmask = ( 1u << Rbits ) - 1u;
            const uint Gmask = ( 1u << Gbits ) - 1u;
            const uint Bmask = ( 1u << Bbits ) - 1u;
            const uint Amask = ( 1u << Abits ) - 1u;
            const uint Gshift = Rbits;
            const uint Bshift = Gshift + Gbits;
            const uint Ashift = Bshift + Bbits;
            const float4 scale = float4( Rmask, Gmask, Bmask, Amask );

            uint4 p = uint4( saturate( c ) * scale + 0.5 );
            p.yzw <<= uint3( Gshift, Bshift, Ashift );
            p.xy |= p.zw;

            return p.x | p.y;
        }


        float4 UintToRgba( uint p,   const uint Rbits,   const uint Gbits = 0,   const uint Bbits = 0,   const uint Abits = 0 )
        {
            const uint Rmask = ( 1u << Rbits ) - 1u;
            const uint Gmask = ( 1u << Gbits ) - 1u;
            const uint Bmask = ( 1u << Bbits ) - 1u;
            const uint Amask = ( 1u << Abits ) - 1u;
            const uint Gshift = Rbits;
            const uint Bshift = Gshift + Gbits;
            const uint Ashift = Bshift + Bbits;
            const float4 scale = 1.0 / max( float4( Rmask, Gmask, Bmask, Amask ), 1.0 );

            uint4 c = p >> uint4( 0, Gshift, Bshift, Ashift );
            c &= uint4( Rmask, Gmask, Bmask, Amask );

            return float4( c ) * scale;
        }


        uint Rg16fToUint( float2 c )
        {
            return ( f32tof16( c.y ) << 16 ) | ( f32tof16( c.x ) & 0xFFFF );
        }

        float2 UintToRg16f( uint p )
        {
            float2 c;
            c.x = f16tof32( p );
            c.y = f16tof32( p >> 16 );

            return c;
        }


        uint EncodeRgbe( float3 c )
        {
            float sharedExp = ceil( log2( max( max( c.x, c.y ), c.z ) ) );
            float4 p = float4( c * exp2( -sharedExp ), ( sharedExp + 128.0 ) / 255.0 );

            return RgbaToUint( p, 8, 8, 8, 8 );
        }

        float3 DecodeRgbe( uint p )
        {
            float4 c = UintToRgba( p, 8, 8, 8, 8 );

            return c.xyz * exp2( c.w * 255.0 - 128.0 );
        }










        float2 EncodeUnitVector( float3 v,   const bool bSigned = false )
        {
            v /= abs( v.x ) + abs( v.y ) + abs( v.z );
            v.xy = v.z >= 0.0 ? v.xy : ( 1.0 - abs( v.yx ) ) * Math::Sign( v.xy );

            return bSigned ? v.xy : saturate( v.xy * 0.5 + 0.5 );
        }

        float3 DecodeUnitVector( float2 p,   const bool bSigned = false,   const bool bNormalize = true )
        {
            p = bSigned ? p : ( p * 2.0 - 1.0 );


            float3 n = float3( p.xy, 1.0 - abs( p.x ) - abs( p.y ) );
            float t = saturate( -n.z );
            n.xy += n.xy >= 0.0 ? -t : t;

            return bNormalize ? normalize( n ) : n;
        }
    };





    namespace Filtering
    {
        float GetModifiedRoughnessFromNormalVariance( float linearRoughness, float3 nonNormalizedAverageNormal )
        {

            float l = length( nonNormalizedAverageNormal );
            float kappa = saturate( 1.0 - l * l ) * Math::PositiveRcp( l * ( 3.0 - l * l ) );

            return Math::Sqrt01( linearRoughness * linearRoughness + kappa );
        }


        float GetMipmapLevel( float duvdxMulTexSizeSq, float duvdyMulTexSizeSq,   const float maxAnisotropy = 1.0 )
        {


            float Pmax = max( duvdxMulTexSizeSq, duvdyMulTexSizeSq );

            if( maxAnisotropy > 1.0 )
            {
                float Pmin = min( duvdxMulTexSizeSq, duvdyMulTexSizeSq );
                float N = min( Pmax * Math::PositiveRcp( Pmin ), maxAnisotropy );
                Pmax *= Math::PositiveRcp( N );
            }

            float mip = 0.5 * log2( Pmax );

            return mip;
        }


        struct Nearest
        {
            float2 origin;
        };

        Nearest GetNearestFilter( float2 uv, float2 texSize )
        {
            float2 t = uv * texSize;

            Nearest result;
            result.origin = floor( t );

            return result;
        }


        struct Bilinear
        {
            float2 origin;
            float2 weights;
        };

        Bilinear GetBilinearFilter( float2 uv, float2 texSize )
        {
            float2 t = uv * texSize - 0.5;

            Bilinear result;
            result.origin = floor( t );
            result.weights = t - result.origin;

            return result;
        }

        float ApplyBilinearFilter( float s00, float s10, float s01, float s11, Bilinear f )
        { return lerp( lerp( s00, s10, f.weights.x ), lerp( s01, s11, f.weights.x ), f.weights.y ); }

        float2 ApplyBilinearFilter( float2 s00, float2 s10, float2 s01, float2 s11, Bilinear f )
        { return lerp( lerp( s00, s10, f.weights.x ), lerp( s01, s11, f.weights.x ), f.weights.y ); }

        float3 ApplyBilinearFilter( float3 s00, float3 s10, float3 s01, float3 s11, Bilinear f )
        { return lerp( lerp( s00, s10, f.weights.x ), lerp( s01, s11, f.weights.x ), f.weights.y ); }

        float4 ApplyBilinearFilter( float4 s00, float4 s10, float4 s01, float4 s11, Bilinear f )
        { return lerp( lerp( s00, s10, f.weights.x ), lerp( s01, s11, f.weights.x ), f.weights.y ); }

        float4 GetBilinearCustomWeights( Bilinear f, float4 customWeights )
        {
            float2 oneMinusWeights = 1.0 - f.weights;

            float4 weights = customWeights;
            weights.x *= oneMinusWeights.x * oneMinusWeights.y;
            weights.y *= f.weights.x * oneMinusWeights.y;
            weights.z *= oneMinusWeights.x * f.weights.y;
            weights.w *= f.weights.x * f.weights.y;

            return weights;
        }



        float ApplyBilinearCustomWeights( float s00, float s10, float s01, float s11, float4 w,   const bool normalize = true )
        { return  ( ( s00 * w.x + s10 * w.y + s01 * w.z + s11 * w.w ) * ( normalize ? Math::PositiveRcp( dot( w, 1.0 ) ) : 1.0 ) ) ; }

        float2 ApplyBilinearCustomWeights( float2 s00, float2 s10, float2 s01, float2 s11, float4 w,   const bool normalize = true )
        { return  ( ( s00 * w.x + s10 * w.y + s01 * w.z + s11 * w.w ) * ( normalize ? Math::PositiveRcp( dot( w, 1.0 ) ) : 1.0 ) ) ; }

        float3 ApplyBilinearCustomWeights( float3 s00, float3 s10, float3 s01, float3 s11, float4 w,   const bool normalize = true )
        { return  ( ( s00 * w.x + s10 * w.y + s01 * w.z + s11 * w.w ) * ( normalize ? Math::PositiveRcp( dot( w, 1.0 ) ) : 1.0 ) ) ; }

        float4 ApplyBilinearCustomWeights( float4 s00, float4 s10, float4 s01, float4 s11, float4 w,   const bool normalize = true )
        { return  ( ( s00 * w.x + s10 * w.y + s01 * w.z + s11 * w.w ) * ( normalize ? Math::PositiveRcp( dot( w, 1.0 ) ) : 1.0 ) ) ; }


        struct CatmullRom
        {
            float2 origin;
            float2 weights[4];
        };

        CatmullRom GetCatmullRomFilter( float2 uv, float2 texSize, float sharpness = 0.5 )
        {
            float2 tci = uv * texSize;
            float2 tc = floor( tci - 0.5 ) + 0.5;
            float2 f = saturate( tci - tc );
            float2 f2 = f * f;
            float2 f3 = f2 * f;

            CatmullRom result;
            result.origin = tc - 1.5;
            result.weights[ 0 ] = -sharpness * f3 + 2.0 * sharpness * f2 - sharpness * f;
            result.weights[ 1 ] = ( 2.0 - sharpness ) * f3 - ( 3.0 - sharpness ) * f2 + 1.0;
            result.weights[ 2 ] = -( 2.0 - sharpness ) * f3 + ( 3.0 - 2.0 * sharpness ) * f2 + sharpness * f;
            result.weights[ 3 ] = sharpness * f3 - sharpness * f2;

            return result;
        }

        float4 ApplyCatmullRomFilterNoCorners( CatmullRom filter, float4 s10, float4 s20, float4 s01, float4 s11, float4 s21, float4 s31, float4 s02, float4 s12, float4 s22, float4 s32, float4 s13, float4 s23 )
        {
#line 1247 "/Plugin/NRD/Private/Reblur/STL.ush"
            float w = filter.weights[ 1 ].x * filter.weights[ 0 ].y;
            float4 color = s10 * w;
            float sum = w;

            w = filter.weights[ 2 ].x * filter.weights[ 0 ].y;
            color += s20 * w;
            sum += w;


            w = filter.weights[ 0 ].x * filter.weights[ 1 ].y;
            color += s01 * w;
            sum += w;

            w = filter.weights[ 1 ].x * filter.weights[ 1 ].y;
            color += s11 * w;
            sum += w;

            w = filter.weights[ 2 ].x * filter.weights[ 1 ].y;
            color += s21 * w;
            sum += w;

            w = filter.weights[ 3 ].x * filter.weights[ 1 ].y;
            color += s31 * w;
            sum += w;


            w = filter.weights[ 0 ].x * filter.weights[ 2 ].y;
            color += s02 * w;
            sum += w;

            w = filter.weights[ 1 ].x * filter.weights[ 2 ].y;
            color += s12 * w;
            sum += w;

            w = filter.weights[ 2 ].x * filter.weights[ 2 ].y;
            color += s22 * w;
            sum += w;

            w = filter.weights[ 3 ].x * filter.weights[ 2 ].y;
            color += s32 * w;
            sum += w;


            w = filter.weights[ 1 ].x * filter.weights[ 3 ].y;
            color += s13 * w;
            sum += w;

            w = filter.weights[ 2 ].x * filter.weights[ 3 ].y;
            color += s23 * w;
            sum += w;

            return color * Math::PositiveRcp( sum );
        }

        float ApplyCatmullRomFilterNoCorners( CatmullRom filter, float s10, float s20, float s01, float s11, float s21, float s31, float s02, float s12, float s22, float s32, float s13, float s23 )
        {
#line 1312 "/Plugin/NRD/Private/Reblur/STL.ush"
            float w = filter.weights[ 1 ].x * filter.weights[ 0 ].y;
            float color = s10 * w;
            float sum = w;

            w = filter.weights[ 2 ].x * filter.weights[ 0 ].y;
            color += s20 * w;
            sum += w;


            w = filter.weights[ 0 ].x * filter.weights[ 1 ].y;
            color += s01 * w;
            sum += w;

            w = filter.weights[ 1 ].x * filter.weights[ 1 ].y;
            color += s11 * w;
            sum += w;

            w = filter.weights[ 2 ].x * filter.weights[ 1 ].y;
            color += s21 * w;
            sum += w;

            w = filter.weights[ 3 ].x * filter.weights[ 1 ].y;
            color += s31 * w;
            sum += w;


            w = filter.weights[ 0 ].x * filter.weights[ 2 ].y;
            color += s02 * w;
            sum += w;

            w = filter.weights[ 1 ].x * filter.weights[ 2 ].y;
            color += s12 * w;
            sum += w;

            w = filter.weights[ 2 ].x * filter.weights[ 2 ].y;
            color += s22 * w;
            sum += w;

            w = filter.weights[ 3 ].x * filter.weights[ 2 ].y;
            color += s32 * w;
            sum += w;


            w = filter.weights[ 1 ].x * filter.weights[ 3 ].y;
            color += s13 * w;
            sum += w;

            w = filter.weights[ 2 ].x * filter.weights[ 3 ].y;
            color += s23 * w;
            sum += w;

            return color * Math::PositiveRcp( sum );
        }



        float4 GetBlurOffsets1D( float2 directionDivTexSize )
        { return float2( 1.7229, 3.8697 ).xxyy * directionDivTexSize.xyxy; }



        float4 GetBlurOffsets2D( float2 invTexSize )
        { return float4( 0.4, 0.9, -0.4, -0.9 ) * invTexSize.xyxy; }
    };





    namespace Sequence
    {





        uint Bayer4x4ui( uint2 samplePos, uint frameIndex,   const uint mode =  1  )
        {
            uint2 samplePosWrap = samplePos & 3;
            uint a = 2068378560 * ( 1 - ( samplePosWrap.x >> 1 ) ) + 1500172770 * ( samplePosWrap.x >> 1 );
            uint b = ( samplePosWrap.y + ( ( samplePosWrap.x & 1 ) << 2 ) ) << 2;

            uint sampleOffset = mode ==  1  ? Math::ReverseBits4( frameIndex ) : frameIndex;

            return ( ( a >> b ) + sampleOffset ) & 0xF;
        }


        float Bayer4x4( uint2 samplePos, uint frameIndex,   const uint mode =  1  )
        {
            uint bayer = Bayer4x4ui( samplePos, frameIndex, mode );

            return float( bayer ) / 16.0;
        }



        float2 Hammersley2D( uint index, float sampleCount )
        {
            float x = float( index ) / sampleCount;
            float y = float( Math::ReverseBits32( index ) ) * 2.3283064365386963e-10;

            return float2( x, y );
        }


        uint2 Morton2D( uint index )
        {
            return uint2( Math::CompactBits( index ), Math::CompactBits( index >> 1 ) );
        }


        uint CheckerBoard( uint2 samplePos, uint frameIndex )
        {
            uint a = samplePos.x ^ samplePos.y;

            return ( a ^ frameIndex ) & 0x1;
        }

        uint IntegerExplode( uint x )
        {
            x = ( x | ( x << 8 ) ) & 0x00FF00FF;
            x = ( x | ( x << 4 ) ) & 0x0F0F0F0F;
            x = ( x | ( x << 2 ) ) & 0x33333333;
            x = ( x | ( x << 1 ) ) & 0x55555555;
            return x;
        }

        uint Zorder( uint2 xy )
        {
            return IntegerExplode( xy.x ) | ( IntegerExplode( xy.y ) << 1 );
        }
    };





    namespace Rng
    {
        static uint2 g_Seed;




        void Initialize( uint2 samplePos, uint frameIndex, uint spinNum = 16 )
        {
            g_Seed.x = Sequence::Zorder( samplePos );
            g_Seed.y = frameIndex;

            uint s = 0;
            [unroll]
            for( uint n = 0; n < spinNum; n++ )
            {
                s += 0x9E3779B9;
                g_Seed.x += ( ( g_Seed.y << 4 ) + 0xA341316C ) ^ ( g_Seed.y + s ) ^ ( ( g_Seed.y >> 5 ) + 0xC8013EA4 );
                g_Seed.y += ( ( g_Seed.x << 4 ) + 0xAD90777D ) ^ ( g_Seed.x + s ) ^ ( ( g_Seed.x >> 5 ) + 0x7E95761E );
            }
        }

        uint2 GetUint2( )
        {





                g_Seed ^= g_Seed << 13;
                g_Seed ^= g_Seed >> 17;
                g_Seed ^= g_Seed << 5;


            return g_Seed;
        }


        float2 GetFloat2(   const uint mode =  1  )
        {
            uint2 r = GetUint2( );

            if( mode ==  1  )
                return 2.0 - asfloat( ( r >> 9 ) | 0x3F800000 );

            return float2( r >> 8 ) * ( 1.0 / float( 1 << 24 ) );
        }

        float4 GetFloat4(   const uint mode =  1  )
        {
            uint4 r;
            r.xy = GetUint2( );
            r.zw = GetUint2( );

            if( mode ==  1  )
                return 2.0 - asfloat( ( r >> 9 ) | 0x3F800000 );

            return float4( r >> 8 ) * ( 1.0 / float( 1 << 24 ) );
        }
    };
#line 1529 "/Plugin/NRD/Private/Reblur/STL.ush"
    namespace BRDF
    {
        float Pow5( float x )
        { return Math::Pow01( 1.0 - x, 5.0 ); }

        void ConvertBaseColorMetalnessToAlbedoRf0( float3 baseColor, float metalness, out float3 albedo, out float3 Rf0 )
        {



            albedo = baseColor * saturate( 1.0 - metalness );
            Rf0 = lerp(  0.04 , baseColor, metalness );
        }

        float FresnelTerm_Shadowing( float3 Rf0 )
        {

            return saturate( Color::Luminance( Rf0 ) / 0.02 );
        }





        float DiffuseTerm_Lambert( float linearRoughness, float NoL, float NoV, float VoH )
        {
            float d = 1.0;

            return d / Math::Pi( 1.0 );
        }


        float DiffuseTerm_Burley( float linearRoughness, float NoL, float NoV, float VoH )
        {
            float f = 2.0 * VoH * VoH * linearRoughness - 0.5;
            float FdV = f * Pow5( NoV ) + 1.0;
            float FdL = f * Pow5( NoL ) + 1.0;
            float d = FdV * FdL;

            return d / Math::Pi( 1.0 );
        }


        float DiffuseTerm_OrenNayar( float linearRoughness, float NoL, float NoV, float VoH )
        {
            float m = linearRoughness * linearRoughness;
            float m2 = m * m;
            float VoL = 2.0 * VoH - 1.0;
            float c1 = 1.0 - 0.5 * m2 / ( m2 + 0.33 );
            float cosri = VoL - NoV * NoL;
            float a = cosri >= 0.0 ? saturate( NoL * Math::PositiveRcp( NoV ) ) : NoL;
            float c2 = 0.45 * m2 / ( m2 + 0.09 ) * cosri * a;
            float d = NoL * c1 + c2;

            return d / Math::Pi( 1.0 );
        }

        float DiffuseTerm( float linearRoughness, float NoL, float NoV, float VoH )
        {




            return DiffuseTerm_Burley( linearRoughness, NoL, NoV, VoH );
        }






        float DistributionTerm_Blinn( float linearRoughness, float NoH )
        {
            float m = linearRoughness * linearRoughness;
            float m2 = m * m;
            float alpha = 2.0 * Math::PositiveRcp( m2 ) - 2.0;
            float norm = ( alpha + 2.0 ) / 2.0;
            float d = norm * Math::Pow01( NoH, alpha );

            return d / Math::Pi( 1.0 );
        }


        float DistributionTerm_Beckmann( float linearRoughness, float NoH )
        {
            float m = linearRoughness * linearRoughness;
            float m2 = m * m;
            float b = NoH * NoH;
            float a = m2 * b;
            float d = exp( ( b - 1.0 ) * Math::PositiveRcp( a ) ) * Math::PositiveRcp( a * b );

            return d / Math::Pi( 1.0 );
        }


        float DistributionTerm_GGX( float linearRoughness, float NoH )
        {
            float m = linearRoughness * linearRoughness;
            float m2 = m * m;


                float t = 1.0 - NoH * NoH * ( 0.99999994 - m2 );
                float a = max( m, 1e-6 ) / t;
#line 1637 "/Plugin/NRD/Private/Reblur/STL.ush"
            float d = a * a;

            return d / Math::Pi( 1.0 );
        }


        float DistributionTerm_GTR( float linearRoughness, float NoH )
        {
            float m = linearRoughness * linearRoughness;
            float m2 = m * m;
            float t = ( NoH * m2 - NoH ) * NoH + 1.0;

            float t1 = Math::Pow01( t, - 1.5  );
            float t2 = 1.0 - Math::Pow01( m2, -(  1.5  - 1.0 ) );
            float d = (  1.5  - 1.0 ) * ( m2 * t1 - t1 ) * Math::PositiveRcp( t2 );

            return d / Math::Pi( 1.0 );
        }

        float DistributionTerm( float linearRoughness, float NoH )
        {





            return DistributionTerm_GGX( linearRoughness, NoH );
        }






        float GeometryTerm_Smith( float linearRoughness, float NoVL )
        {
            float m = linearRoughness * linearRoughness;
            float m2 = m * m;
            float a = NoVL + Math::Sqrt01( ( NoVL - m2 * NoVL ) * NoVL + m2 );

            return 2.0 * NoVL * Math::PositiveRcp( a );
        }






        float GeometryTermMod_Implicit( float linearRoughness, float NoL, float NoV, float VoH, float NoH )
        {
            return 0.25;
        }


        float GeometryTermMod_Neumann( float linearRoughness, float NoL, float NoV, float VoH, float NoH )
        {
            return 0.25 * Math::PositiveRcp( max( NoL, NoV ) );
        }


        float GeometryTermMod_Schlick( float linearRoughness, float NoL, float NoV, float VoH, float NoH )
        {
            float m = linearRoughness * linearRoughness;





            float k = m * 0.5;

            float a = NoL * ( 1.0 - k ) + k;
            float b = NoV * ( 1.0 - k ) + k;

            return 0.25 / max( a * b, 1e-6 );
        }




        float GeometryTermMod_SmithCorrelated( float linearRoughness, float NoL, float NoV, float VoH, float NoH )
        {
            float m = linearRoughness * linearRoughness;
            float m2 = m * m;
            float a = NoV * Math::Sqrt01( ( NoL - m2 * NoL ) * NoL + m2 );
            float b = NoL * Math::Sqrt01( ( NoV - m2 * NoV ) * NoV + m2 );

            return 0.5 * Math::PositiveRcp( a + b );
        }




        float GeometryTermMod_SmithUncorrelated( float linearRoughness, float NoL, float NoV, float VoH, float NoH )
        {
            float m = linearRoughness * linearRoughness;
            float m2 = m * m;
            float a = NoL + Math::Sqrt01( ( NoL - m2 * NoL ) * NoL + m2 );
            float b = NoV + Math::Sqrt01( ( NoV - m2 * NoV ) * NoV + m2 );

            return Math::PositiveRcp( a * b );
        }


        float GeometryTermMod_CookTorrance( float linearRoughness, float NoL, float NoV, float VoH, float NoH )
        {
            float k = 2.0 * NoH / VoH;
            float a = min( k * NoV, k * NoL );

            return saturate( a ) * 0.25 * Math::PositiveRcp( NoV * NoL );
        }

        float GeometryTermMod( float linearRoughness, float NoL, float NoV, float VoH, float NoH )
        {







            return GeometryTermMod_SmithCorrelated( linearRoughness, NoL, NoV, VoH, NoH );
        }





        float3 FresnelTerm_None( float3 Rf0, float VoNH )
        {
            return Rf0;
        }


        float3 FresnelTerm_Schlick( float3 Rf0, float VoNH )
        {
            return Rf0 + ( 1.0 - Rf0 ) * Pow5( VoNH );
        }

        float3 FresnelTerm_Fresnel( float3 Rf0, float VoNH )
        {
            float3 nu = Math::Sqrt01( Rf0 );
            nu = ( 1.0 + nu ) * Math::PositiveRcp( 1.0 - nu );

            float k = VoNH * VoNH - 1.0;
            float3 g = sqrt( nu * nu + k );
            float3 a = ( g - VoNH ) / ( g + VoNH );
            float3 c = ( g * VoNH + k ) / ( g * VoNH - k );

            return 0.5 * a * a * ( c * c + 1.0 );
        }

        float3 FresnelTerm( float3 Rf0, float VoNH )
        {




            return FresnelTerm_Schlick( Rf0, VoNH ) * FresnelTerm_Shadowing( Rf0 );
        }







        float3 EnvironmentTerm_Pesce( float3 Rf0, float NoV, float linearRoughness )
        {
            float m = linearRoughness * linearRoughness;

            float a = 7.0 * NoV + 4.0 * m;
            float bias = exp2( -a );

            float b = min( linearRoughness, 0.739 + 0.323 * NoV ) - 0.434;
            float scale = 1.0 - bias - m * max( bias, b );

            bias *= FresnelTerm_Shadowing( Rf0 );

            return saturate( Rf0 * scale + bias );
        }



        float3 EnvironmentTerm_Ross( float3 Rf0, float NoV, float linearRoughness )
        {
            float m = linearRoughness * linearRoughness;

            float f = Math::Pow01( 1.0 - NoV, 5.0 * exp( -2.69 * m ) ) / ( 1.0 + 22.7 * Math::Pow01( m, 1.5 ) );

            float scale = 1.0 - f;
            float bias = f;

            bias *= FresnelTerm_Shadowing( Rf0 );

            return saturate( Rf0 * scale + bias );
        }


        float3 EnvironmentTerm_Unknown(float3 Rf0, float NoV, float roughness)
        {
            float m = roughness * roughness;

            float4 X;
            X.x = 1.0;
            X.y = NoV;
            X.z = NoV * NoV;
            X.w = NoV * X.z;

            float4 Y;
            Y.x = 1.0;
            Y.y = m;
            Y.z = m * m;
            Y.w = m * Y.z;

            float2x2 M1 = float2x2( 0.99044, -1.28514, 1.29678, -0.755907 );
            float3x3 M2 = float3x3( 1.0, 2.92338, 59.4188, 20.3225, -27.0302, 222.592, 121.563, 626.13, 316.627 );

            float2x2 M3 = float2x2( 0.0365463, 3.32707, 9.0632, -9.04756 );
            float3x3 M4 = float3x3( 1.0, 3.59685, -1.36772, 9.04401, -16.3174, 9.22949, 5.56589, 19.7886, -20.2123 );

            float bias = dot( mul( M1, X.xy ), Y.xy ) * Math::PositiveRcp( dot( mul( M2, X.xyw ), Y.xyw ) );
            float scale = dot( mul( M3, X.xy ), Y.xy ) * Math::PositiveRcp( dot( mul( M4, X.xzw ), Y.xyw ) );

            bias *= FresnelTerm_Shadowing( Rf0 );

            return saturate( Rf0 * scale + bias );
        }

        float3 EnvironmentTerm( float3 Rf0, float NoV, float linearRoughness )
        {




            return EnvironmentTerm_Ross( Rf0, NoV, linearRoughness );
        }





        void DirectLighting( float3 N, float3 L, float3 V, float3 Rf0, float linearRoughness, out float3 Cdiff, out float3 Cspec )
        {
            float3 H = normalize( L + V );

            float NoL = saturate( dot( N, L ) );
            float NoH = saturate( dot( N, H ) );
            float VoH = saturate( dot( V, H ) );


            float NoV = abs( dot( N, V ) );

            float D = DistributionTerm( linearRoughness, NoH );
            float G = GeometryTermMod( linearRoughness, NoL, NoV, VoH, NoH );
            float3 F = FresnelTerm( Rf0, VoH );
            float Kdiff = DiffuseTerm( linearRoughness, NoL, NoV, VoH );

            Cspec = F * D * G * NoL;
            Cdiff = ( 1.0 - F ) * Kdiff * NoL;

            Cspec = saturate( Cspec );
        }
    };





    namespace ImportanceSampling
    {

        float GetSpecularLobeHalfAngle( float linearRoughness, float percentOfVolume = 0.75 )
        {
            float m = linearRoughness * linearRoughness;






                return atan( m * percentOfVolume / ( 1.0 - percentOfVolume ) );
#line 1921 "/Plugin/NRD/Private/Reblur/STL.ush"
        }

        float3 CorrectDirectionToInfiniteSource( float3 N, float3 L, float3 V, float tanOfAngularSize )
        {
            float3 R = reflect( -V, N );
            float3 centerToRay = L - dot( L, R ) * R;
            float3 closestPoint = centerToRay * saturate( tanOfAngularSize * Math::Rsqrt( Math::LengthSquared( centerToRay ) ) );

            return normalize( L - closestPoint );
        }






        float GetSpecularDominantFactor( float NoV, float linearRoughness,   const uint mode =  2  )
        {
            float dominantFactor;
            if( mode ==  1  )
            {
                float a = 0.298475 * log( 39.4115 - 39.0029 * linearRoughness );
                dominantFactor = Math::Pow01( 1.0 - NoV, 10.8649 ) * ( 1.0 - a ) + a;
            }
            else if( mode ==  0  )
                dominantFactor = 0.298475 * NoV * log( 39.4115 - 39.0029 * linearRoughness ) + ( 0.385503 - 0.385503 * NoV ) * log( 13.1567 - 12.2848 * linearRoughness );
            else
            {
                float s = 1.0 - linearRoughness;
                dominantFactor = s * ( Math::Sqrt01( s ) + linearRoughness );
            }

            return saturate( dominantFactor );
        }

        float3 GetSpecularDominantDirectionWithFactor( float3 N, float3 V, float dominantFactor )
        {
            float3 R = reflect( -V, N );
            float3 D = lerp( N, R, dominantFactor );

            return normalize( D );
        }

        float4 GetSpecularDominantDirection( float3 N, float3 V, float linearRoughness,   const uint mode =  2  )
        {
            float NoV = abs( dot( N, V ) );
            float dominantFactor = GetSpecularDominantFactor( NoV, linearRoughness, mode );

            return float4( GetSpecularDominantDirectionWithFactor( N, V, dominantFactor ), dominantFactor );
        }





        namespace Uniform
        {
            float GetPDF( )
            {
                return 1.0 / Math::Pi( 2.0 );
            }

            float3 GetRay( float2 rnd )
            {
                float cosTheta = rnd.y;

                float sinTheta = Math::Sqrt01( 1.0 - cosTheta * cosTheta );
                float phi = rnd.x * Math::Pi( 2.0 );

                float3 ray;
                ray.x = sinTheta * cos( phi );
                ray.y = sinTheta * sin( phi );
                ray.z = cosTheta;

                return ray;
            }
        }





        namespace Cosine
        {
            float GetPDF( float NoL = 1.0 )
            {
                float pdf = NoL / Math::Pi( 1.0 );

                return max( pdf, 1e-7 );
            }

            float3 GetRay( float2 rnd )
            {
                float cosTheta = Math::Sqrt01( rnd.y );

                float sinTheta = Math::Sqrt01( 1.0 - cosTheta * cosTheta );
                float phi = rnd.x * Math::Pi( 2.0 );

                float3 ray;
                ray.x = sinTheta * cos( phi );
                ray.y = sinTheta * sin( phi );
                ray.z = cosTheta;

                return ray;
            }
        }





        namespace NDF
        {
            float GetPDF( float linearRoughness, float NoH, float VoH )
            {
                float pdf = BRDF::DistributionTerm_GGX( linearRoughness, NoH );
                pdf *= NoH;
                pdf *= Math::PositiveRcp( 4.0 * VoH );

                return max( pdf, 1e-7 );
            }

            float3 GetRay( float2 rnd, float linearRoughness )
            {
                float m = linearRoughness * linearRoughness;
                float m2 = m * m;
                float t = ( m2 - 1.0 ) * rnd.y + 1.0;
                float cosThetaSq = ( 1.0 - rnd.y ) * Math::PositiveRcp( t );
                float sinTheta = Math::Sqrt01( 1.0 - cosThetaSq );
                float phi = rnd.x * Math::Pi( 2.0 );

                float3 ray;
                ray.x = sinTheta * cos( phi );
                ray.y = sinTheta * sin( phi );
                ray.z = Math::Sqrt01( cosThetaSq );

                return ray;
            }
        }






        namespace VNDF
        {
            float GetPDF( float NoV, float NoH, float linearRoughness )
            {
#line 2086 "/Plugin/NRD/Private/Reblur/STL.ush"
                float m = linearRoughness * linearRoughness;
                float m2 = m * m;
                float a = NoV + Math::Sqrt01( ( NoV - m2 * NoV ) * NoV + m2 );

                float pdf = Math::PositiveRcp( a );
                pdf *= BRDF::DistributionTerm_GGX( linearRoughness, NoH );
                pdf *= 0.5;

                return max( pdf, 1e-7 );
            }

            float3 GetRay( float2 rnd, float2 linearRoughness, float3 Vlocal, float trimFactor = 1.0 )
            {
                const float EPS = 1e-7;




                float2 m = linearRoughness * linearRoughness;


                float3 Vh = normalize( float3( m * Vlocal.xy, Vlocal.z ) );


                float lensq = dot( Vh.xy, Vh.xy );
                float3 T1 = lensq > EPS ? float3( -Vh.y, Vh.x, 0.0 ) * rsqrt( lensq ) : float3( 1.0, 0.0, 0.0 );
                float3 T2 = cross( Vh, T1 );



                float r = Math::Sqrt01( rnd.x * trimFactor );
                float phi = rnd.y * Math::Pi( 2.0 );
                float t1 = r * cos( phi );
                float t2 = r * sin( phi );
                float s = 0.5 * ( 1.0 + Vh.z );
                t2 = ( 1.0 - s ) * Math::Sqrt01( 1.0 - t1 * t1 ) + s * t2;


                float3 Nh = t1 * T1 + t2 * T2 + Math::Sqrt01( 1.0 - t1 * t1 - t2 * t2 ) * Vh;


                float3 Ne = normalize( float3( m * Nh.xy, max( Nh.z, EPS ) ) );

                return Ne;
            }
        }
    };





    struct SH1
    {
        float c0;
        float3 c1;
        float2 chroma;
    };

    namespace SphericalHarmonics
    {
        SH1 ConvertToSecondOrder( float3 color, float3 direction )
        {
            float3 YCoCg = Color::LinearToYCoCg( color );

            SH1 sh;
            sh.c0 = 0.282095 * YCoCg.x;
            sh.c1 = 0.488603 * YCoCg.x * direction;
            sh.chroma = YCoCg.yz;

            return sh;
        }

        float3 ExtractColor( SH1 sh )
        {
            float Y = sh.c0 / 0.282095;

            return Color::YCoCgToLinear( float3( Y, sh.chroma.x, sh.chroma.y ) );
        }

        float3 ExtractDirection( SH1 sh )
        {
            return sh.c1 * Math::Rsqrt( Math::LengthSquared( sh.c1 ) );
        }


        float3 ResolveColorToDiffuse( SH1 sh, float3 N, float cosHalfAngle = 0.0 )
        {
            float d = dot( sh.c1, N );
            float Y = 1.023326 * d + 0.886226 * sh.c0;


            Y = max( Y, 0 );


            Y *= Geometry::SolidAngle( cosHalfAngle ) / Math::Pi( 1.0 );


            float modifier = 0.282095 * Y * Math::PositiveRcp( sh.c0 );
            float2 CoCg = sh.chroma * saturate( modifier );

            return Color::YCoCgToLinear( float3( Y, CoCg.x, CoCg.y ) );
        }
    };
}
#line 14 "/Plugin/NRD/Private/Reblur/REBLUR_DiffuseOcclusion_TemporalAccumulation.cs.usf"
#line 1 "NRD.ush"
#line 223 "/Plugin/NRD/Private/Reblur/NRD.ush"
float _NRD_PackViewZ( float z )
{
    return clamp( z *  0.125 , - 65504.0 ,  65504.0  );
}


float2 _NRD_EncodeUnitVector( float3 v, const bool bSigned = false )
{
    v /= dot( abs( v ), 1.0 );

    float2 octWrap = ( 1.0 - abs( v.yx ) ) * ( v.xy >= 0.0 ? 1.0 : -1.0 );
    v.xy = v.z >= 0.0 ? v.xy : octWrap;

    return bSigned ? v.xy : v.xy * 0.5 + 0.5;
}

float3 _NRD_DecodeUnitVector( float2 p, const bool bSigned = false, const bool bNormalize = true )
{
    p = bSigned ? p : ( p * 2.0 - 1.0 );


    float3 n = float3( p.xy, 1.0 - abs( p.x ) - abs( p.y ) );
    float t = saturate( -n.z );
    n.xy += n.xy >= 0.0 ? -t : t;

    return bNormalize ? normalize( n ) : n;
}


float _NRD_Luminance( float3 linearColor )
{
    return dot( linearColor, float3( 0.2990, 0.5870, 0.1140 ) );
}

float3 _NRD_LinearToYCoCg( float3 color )
{
    float Y = dot( color, float3( 0.25, 0.5, 0.25 ) );
    float Co = dot( color, float3( 0.5, 0.0, -0.5 ) );
    float Cg = dot( color, float3( -0.25, 0.5, -0.25 ) );

    return float3( Y, Co, Cg );
}

float3 _NRD_YCoCgToLinear( float3 color )
{
    float t = color.x - color.z;

    float3 r;
    r.y = color.x + color.z;
    r.x = t + color.y;
    r.z = t - color.y;

    return max( r, 0.0 );
}


float _REBLUR_GetHitDistanceNormalization( float viewZ, float4 hitDistParams, float roughness = 1.0 )
{
    return ( hitDistParams.x + abs( viewZ ) * hitDistParams.y ) * lerp( 1.0, hitDistParams.z, saturate( exp2( hitDistParams.w * roughness * roughness ) ) );
}






struct NRD_SH
{
    float3 c0_chroma;
    float3 c1;
    float normHitDist;
};

NRD_SH NRD_SH_Create( float3 color, float3 direction )
{
    float3 YCoCg = _NRD_LinearToYCoCg( color );

    NRD_SH sh;
    sh.c0_chroma = 0.282095 * YCoCg;
    sh.c1 = 0.488603 * YCoCg.x * direction;

    return sh;
}

float3 NRD_SH_ExtractColor( NRD_SH sh )
{
    return _NRD_YCoCgToLinear( sh.c0_chroma / 0.282095 );
}

float3 NRD_SH_ExtractDirection( NRD_SH sh )
{
    return sh.c1 * rsqrt( dot( sh.c1, sh.c1 ) + 1e-7 );
}

void NRD_SH_Add( inout NRD_SH result, NRD_SH x )
{
    result.c0_chroma += x.c0_chroma;
    result.c1 += x.c1;
}

void NRD_SH_Mul( inout NRD_SH result, float x )
{
    result.c0_chroma *= x;
    result.c1 *= x;
}


float3 NRD_SH_ResolveColor( NRD_SH sh, float3 N, float cosHalfAngle = 0.0 )
{
    float d = dot( N, sh.c1 );
    float Y = 1.023326 * max( d, 0.0 ) + 0.886226 * max( sh.c0_chroma.x, 0.0 );


    float solidAngle = 2.0 *  3.14159265358979323846  * ( 1.0 - cosHalfAngle );
    Y *= solidAngle /  3.14159265358979323846 ;


    float eps = 1e-6;
    float modifier = ( Y + eps ) / ( sh.c0_chroma.x + eps );
    float2 CoCg = sh.c0_chroma.yz * modifier;

    return _NRD_YCoCgToLinear( float3( Y, CoCg ) );
}
#line 357 "/Plugin/NRD/Private/Reblur/NRD.ush"
float4 NRD_FrontEnd_UnpackNormalAndRoughness( float4 p, out float materialID )
{
    materialID = 0;

    float4 r;








        r.xyz = p.xyz * 2.0 - 1.0;
        r.w = p.w;


    r.xyz = normalize( r.xyz );
#line 380 "/Plugin/NRD/Private/Reblur/NRD.ush"
    return r;
}


float4 NRD_FrontEnd_UnpackNormalAndRoughness( float4 p )
{
    float unused;

    return NRD_FrontEnd_UnpackNormalAndRoughness( p, unused );
}



float4 NRD_FrontEnd_PackNormalAndRoughness( float3 N, float roughness, uint materialID = 0 )
{
    float4 p;
#line 407 "/Plugin/NRD/Private/Reblur/NRD.ush"
        N /= max( abs( N.x ), max( abs( N.y ), abs( N.z ) ) );

        p.xyz = N * 0.5 + 0.5;
        p.w = roughness;


    return p;
}




float4 NRD_FrontEnd_PackDirectionAndPdf( float3 direction, float pdf )
{
    return float4( direction, pdf );
}



float4 NRD_FrontEnd_UnpackDirectionAndPdf( float4 directionAndPdf )
{
    return directionAndPdf;
}






float REBLUR_FrontEnd_GetNormHitDist( float hitDist, float viewZ, float4 hitDistParams, float roughness = 1.0 )
{
    float f = _REBLUR_GetHitDistanceNormalization( viewZ, hitDistParams, roughness );

    return saturate( hitDist / f );
}




float4 REBLUR_FrontEnd_PackRadianceAndNormHitDist( float3 radiance, float normHitDist, bool sanitize = true )
{
    if( sanitize )
    {
        radiance = any( isnan( radiance ) | isinf( radiance ) ) ? 0 : clamp( radiance, 0,  65504.0  );
        normHitDist = ( isnan( normHitDist ) | isinf( normHitDist ) ) ? 0 : saturate( normHitDist );
    }


    if( normHitDist != 0 )
        normHitDist = max( normHitDist,  1e-7  );

    radiance = _NRD_LinearToYCoCg( radiance );

    return float4( radiance, normHitDist );
}



float4 REBLUR_FrontEnd_PackDirectionAndNormHitDist( float3 direction, float normHitDist, bool sanitize = true )
{
    if( sanitize )
    {
        direction = any( isnan( direction ) | isinf( direction ) ) ? 0 : direction;
        normHitDist = ( isnan( normHitDist ) | isinf( normHitDist ) ) ? 0 : saturate( normHitDist );
    }


    if( normHitDist != 0 )
        normHitDist = max( normHitDist,  1e-7  );

    return float4( direction * 0.5 + 0.5, normHitDist );
}




float4 REBLUR_FrontEnd_PackSh( float3 radiance, float normHitDist, float3 direction, float pdf, out float4 out1, bool sanitize = true )
{
    if( sanitize )
    {
        radiance = any( isnan( radiance ) | isinf( radiance ) ) ? 0 : clamp( radiance, 0,  65504.0  );
        normHitDist = ( isnan( normHitDist ) | isinf( normHitDist ) ) ? 0 : saturate( normHitDist );
    }


    if( normHitDist != 0 )
        normHitDist = max( normHitDist,  1e-7  );

    NRD_SH sh = NRD_SH_Create( radiance, direction );


    float4 out0 = float4( sh.c0_chroma, normHitDist );


    out1 = float4( sh.c1, pdf );

    return out0;
}







float4 RELAX_FrontEnd_PackRadianceAndHitDist( float3 radiance, float hitDist, bool sanitize = true )
{
    if( sanitize )
    {
        radiance = any( isnan( radiance ) | isinf( radiance ) ) ? 0 : clamp( radiance, 0,  65504.0  );
        hitDist = ( isnan( hitDist ) | isinf( hitDist ) ) ? 0 : clamp( hitDist, 0,  65504.0  );
    }


    if( hitDist != 0 )
        hitDist = max( hitDist,  1e-7  );

    return float4( radiance, hitDist );
}










float2 SIGMA_FrontEnd_PackShadow( float viewZ, float distanceToOccluder, float tanOfLightAngularRadius )
{
    float2 r;
    r.x = 0.0;
    r.y = _NRD_PackViewZ( viewZ );

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


float2 SIGMA_FrontEnd_PackShadow( float viewZ, float distanceToOccluder, float tanOfLightAngularRadius, float3 translucency, out float4 out2 )
{

    out2.x = float( distanceToOccluder ==  65504.0  );
    out2.yzw = saturate( translucency );


    float2 out1 = SIGMA_FrontEnd_PackShadow( viewZ, distanceToOccluder, tanOfLightAngularRadius );

    return out1;
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


float2 SIGMA_FrontEnd_MultiLightEnd( float viewZ,  float2x3  multiLightShadowData, float3 Lsum, out float4 out2 )
{

    out2.yzw = multiLightShadowData[ 0 ] / max( Lsum, 1e-6 );
    out2.x = _NRD_Luminance( out2.yzw );


    float2 out1;
    out1.x = multiLightShadowData[ 1 ].x / max( multiLightShadowData[ 1 ].y, 1e-6 );
    out1.y = _NRD_PackViewZ( viewZ );

    return out1;
}
#line 615 "/Plugin/NRD/Private/Reblur/NRD.ush"
float4 REBLUR_BackEnd_UnpackRadianceAndNormHitDist( float4 color )
{
    color.xyz = _NRD_YCoCgToLinear( color.xyz );

    return color;
}


float4 REBLUR_BackEnd_UnpackDirectionAndNormHitDist( float4 color )
{
    color.xyz = color.xyz * 2.0 - 1.0;

    return color;
}



NRD_SH REBLUR_BackEnd_UnpackSh( float4 data0, float4 data1 )
{
    NRD_SH sh;
    sh.c0_chroma = data0.xyz;
    sh.c1 = data1.xyz;
    sh.normHitDist = data0.w;

    return sh;
}







float4 RELAX_BackEnd_UnpackRadiance( float4 color )
{
    return color;
}
#line 673 "/Plugin/NRD/Private/Reblur/NRD.ush"
float NRD_GetTrimmingFactor( float roughness, float3 trimmingParams )
{
    float trimmingFactor = trimmingParams.x * smoothstep( trimmingParams.y, trimmingParams.z, roughness );

    return trimmingFactor;
}


float NRD_GetSampleWeight( float3 radiance )
{
    return any( isnan( radiance ) | isinf( radiance ) ) ? 0.0 : 1.0;
}


float REBLUR_GetHitDist( float normHitDist, float viewZ, float4 hitDistParams, float roughness )
{
    float scale = _REBLUR_GetHitDistanceNormalization( viewZ, hitDistParams, roughness );

    return normHitDist * scale;
}
#line 15 "/Plugin/NRD/Private/Reblur/REBLUR_DiffuseOcclusion_TemporalAccumulation.cs.usf"
#line 19 "/Plugin/NRD/Private/Reblur/REBLUR_DiffuseOcclusion_TemporalAccumulation.cs.usf"
#line 1 "REBLUR_Config.ush"
#line 32 "/Plugin/NRD/Private/Reblur/REBLUR_Config.ush"
    static const float3 g_Special6[ 6 ] =
    {
        float3( -0.50 * sqrt(3.0) , -0.50 , 1.0 ),
        float3( 0.00 , 1.00 , 1.0 ),
        float3( 0.50 * sqrt(3.0) , -0.50 , 1.0 ),
        float3( 0.00 , -0.30 , 0.3 ),
        float3( 0.15 * sqrt(3.0) , 0.15 , 0.3 ),
        float3( -0.15 * sqrt(3.0) , 0.15 , 0.3 ),
    };


    static const float3 g_Special8[ 8 ] =
    {
        float3( -1.00 , 0.00 , 1.0 ),
        float3( 0.00 , 1.00 , 1.0 ),
        float3( 1.00 , 0.00 , 1.0 ),
        float3( 0.00 , -1.00 , 1.0 ),
        float3( -0.25 * sqrt(2.0) , 0.25 * sqrt(2.0) , 0.5 ),
        float3( 0.25 * sqrt(2.0) , 0.25 * sqrt(2.0) , 0.5 ),
        float3( 0.25 * sqrt(2.0) , -0.25 * sqrt(2.0) , 0.5 ),
        float3( -0.25 * sqrt(2.0) , -0.25 * sqrt(2.0) , 0.5 )
    };
#line 20 "/Plugin/NRD/Private/Reblur/REBLUR_DiffuseOcclusion_TemporalAccumulation.cs.usf"
#line 1 "REBLUR_DiffuseSpecular_TemporalAccumulation.resources.ush"
#line 11 "/Plugin/NRD/Private/Reblur/REBLUR_DiffuseSpecular_TemporalAccumulation.resources.ush"

    SamplerState gNearestClamp;
    SamplerState gNearestMirror;
    SamplerState gLinearClamp;
    SamplerState gLinearMirror;



                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       float gMaxFastAccumulatedFrameNum;                                                                                                                                                                                                                                                                                                                                                     
                              
                              
                          
                               
                        
                        
                    
                              
                                         
                                 
                           
                           
                           

#line 88 "/Plugin/NRD/Private/Reblur/REBLUR_DiffuseSpecular_TemporalAccumulation.resources.ush"

        Texture2D<float4> gIn_Normal_Roughness;
        Texture2D<float> gIn_ViewZ;
        Texture2D<float3> gIn_ObjectMotion;
        Texture2D<float> gIn_Prev_ViewZ;
        Texture2D<float4> gIn_Prev_Normal_Roughness;
        Texture2D<float> gIn_Prev_AccumSpeeds_MaterialID;
        Texture2D<float> gIn_Diff_Confidence;
        Texture2D< float > gIn_Diff;
        Texture2D< float > gIn_Diff_History;
#line 105 "/Plugin/NRD/Private/Reblur/REBLUR_DiffuseSpecular_TemporalAccumulation.resources.ush"



        RWTexture2D< float2 > gOut_Diff;
        RWTexture2D<float4> gOut_Data1;
#line 117 "/Plugin/NRD/Private/Reblur/REBLUR_DiffuseSpecular_TemporalAccumulation.resources.ush"

#line 21 "/Plugin/NRD/Private/Reblur/REBLUR_DiffuseOcclusion_TemporalAccumulation.cs.usf"
#line 22 "/Plugin/NRD/Private/Reblur/REBLUR_DiffuseOcclusion_TemporalAccumulation.cs.usf"
#line 1 "Common.ush"
#line 11 "/Plugin/NRD/Private/Reblur/Common.ush"
#line 1 "Poisson.ush"
#line 40 "/Plugin/NRD/Private/Reblur/Poisson.ush"
static const float3 g_Poisson8[8] =
{
    float3( -0.4706069, -0.4427112, +0.6461146 ),
    float3( -0.9057375, +0.3003471, +0.9542373 ),
    float3( -0.3487388, +0.4037880, +0.5335386 ),
    float3( +0.1023042, +0.6439373, +0.6520134 ),
    float3( +0.5699277, +0.3513750, +0.6695386 ),
    float3( +0.2939128, -0.1131226, +0.3149309 ),
    float3( +0.7836658, -0.4208784, +0.8895339 ),
    float3( +0.1564120, -0.8198990, +0.8346850 )
};


static const float3 g_Poisson16[16] =
{
    float3( -0.0936476, -0.7899283, +0.7954600 ),
    float3( -0.1209752, -0.2627860, +0.2892948 ),
    float3( -0.5646901, -0.7059856, +0.9040413 ),
    float3( -0.8277994, -0.1538168, +0.8419688 ),
    float3( -0.4620740, +0.1951437, +0.5015910 ),
    float3( -0.7517998, +0.5998214, +0.9617633 ),
    float3( -0.0812514, +0.2904110, +0.3015631 ),
    float3( -0.2397440, +0.7581663, +0.7951688 ),
    float3( +0.2446934, +0.9202285, +0.9522055 ),
    float3( +0.4943011, +0.5736654, +0.7572486 ),
    float3( +0.3415412, +0.1412707, +0.3696049 ),
    float3( +0.8744238, +0.3246290, +0.9327384 ),
    float3( +0.7406740, -0.1434729, +0.7544418 ),
    float3( +0.3658852, -0.3596551, +0.5130534 ),
    float3( +0.7880974, -0.5802425, +0.9786618 ),
    float3( +0.3776688, -0.7620423, +0.8504953 )
};


static const float3 g_Poisson32[32] =
{
    float3( -0.1078042, -0.6434212, +0.6523899 ),
    float3( -0.1141091, -0.9539828, +0.9607830 ),
    float3( -0.1982531, -0.3867292, +0.4345846 ),
    float3( -0.5254982, -0.6604451, +0.8440000 ),
    float3( -0.1820032, -0.0936076, +0.2046645 ),
    float3( -0.4654744, -0.2629388, +0.5346057 ),
    float3( -0.7419540, -0.4592809, +0.8726023 ),
    float3( -0.7180300, -0.1888005, +0.7424370 ),
    float3( -0.9541028, -0.0789064, +0.9573601 ),
    float3( -0.6718881, +0.1745270, +0.6941854 ),
    float3( -0.3968981, +0.1973703, +0.4432642 ),
    float3( -0.8614085, +0.4183342, +0.9576158 ),
    float3( -0.5961362, +0.6559430, +0.8863631 ),
    float3( -0.0866527, +0.2057932, +0.2232925 ),
    float3( -0.3287578, +0.7094890, +0.7819567 ),
    float3( -0.0408453, +0.5730602, +0.5745140 ),
    float3( -0.0678108, +0.8920295, +0.8946033 ),
    float3( +0.2702191, +0.9020523, +0.9416564 ),
    float3( +0.2961993, +0.4006296, +0.4982350 ),
    float3( +0.5824130, +0.7839746, +0.9766376 ),
    float3( +0.6095408, +0.4801217, +0.7759233 ),
    float3( +0.5025840, +0.2096348, +0.5445525 ),
    float3( +0.2740403, +0.0734566, +0.2837146 ),
    float3( +0.9130731, +0.4032195, +0.9981425 ),
    float3( +0.7560658, +0.1432026, +0.7695079 ),
    float3( +0.6737013, -0.1910683, +0.7002717 ),
    float3( +0.8628370, -0.3914889, +0.9474974 ),
    float3( +0.7032576, -0.5988359, +0.9236751 ),
    float3( +0.4578032, -0.4541197, +0.6448321 ),
    float3( +0.1706552, -0.3115532, +0.3552304 ),
    float3( +0.2061829, -0.5709705, +0.6070574 ),
    float3( +0.3269635, -0.9024802, +0.9598832 )
};


static const float3 g_Poisson64[64] =
{
    float3( -0.0065114, -0.1460582, +0.1462033 ),
    float3( -0.0303039, -0.9686066, +0.9690805 ),
    float3( -0.1029292, -0.8030527, +0.8096222 ),
    float3( -0.1531820, -0.6213900, +0.6399924 ),
    float3( -0.3230599, -0.8868585, +0.9438674 ),
    float3( -0.1951447, -0.3146919, +0.3702870 ),
    float3( -0.3462451, -0.6440054, +0.7311831 ),
    float3( -0.3455329, -0.4411035, +0.5603260 ),
    float3( -0.6277606, -0.6978221, +0.9386368 ),
    float3( -0.6238620, -0.4722686, +0.7824586 ),
    float3( -0.3958989, -0.2521870, +0.4693977 ),
    float3( -0.8186533, -0.4641639, +0.9410852 ),
    float3( -0.6481082, -0.2896534, +0.7098897 ),
    float3( -0.9109314, -0.1374674, +0.9212455 ),
    float3( -0.6602813, -0.0511829, +0.6622621 ),
    float3( -0.3327182, -0.0034168, +0.3327357 ),
    float3( -0.9708222, +0.0864033, +0.9746596 ),
    float3( -0.7995708, +0.1496022, +0.8134459 ),
    float3( -0.4509301, +0.1788653, +0.4851090 ),
    float3( -0.1161801, +0.0573019, +0.1295427 ),
    float3( -0.6471452, +0.2481229, +0.6930814 ),
    float3( -0.8052469, +0.4099220, +0.9035810 ),
    float3( -0.4898830, +0.3552727, +0.6051480 ),
    float3( -0.6336213, +0.4714487, +0.7897720 ),
    float3( -0.6885121, +0.7122980, +0.9906651 ),
    float3( -0.4522108, +0.5375718, +0.7024800 ),
    float3( -0.1841745, +0.2540318, +0.3137712 ),
    float3( -0.2724991, +0.5243348, +0.5909169 ),
    float3( -0.3906980, +0.8645544, +0.9487356 ),
    float3( -0.1517160, +0.7061030, +0.7222183 ),
    float3( -0.1148268, +0.9200021, +0.9271403 ),
    float3( -0.0228051, +0.5112054, +0.5117138 ),
    float3( +0.0387527, +0.6830538, +0.6841522 ),
    float3( +0.0556644, +0.3292533, +0.3339255 ),
    float3( +0.1651443, +0.8762763, +0.8917022 ),
    float3( +0.3430057, +0.7856857, +0.8572952 ),
    float3( +0.3516012, +0.5249697, +0.6318359 ),
    float3( +0.2562977, +0.3190902, +0.4092762 ),
    float3( +0.5771080, +0.7862252, +0.9752967 ),
    float3( +0.6529276, +0.6084227, +0.8924643 ),
    float3( +0.5189329, +0.4425537, +0.6820155 ),
    float3( +0.8118719, +0.4586847, +0.9324846 ),
    float3( +0.3119081, +0.1337896, +0.3393911 ),
    float3( +0.5046800, +0.1606769, +0.5296404 ),
    float3( +0.6844428, +0.2401899, +0.7253641 ),
    float3( +0.8718888, +0.2715452, +0.9131960 ),
    float3( +0.1815740, +0.0086135, +0.1817782 ),
    float3( +0.9897170, +0.1209020, +0.9970742 ),
    float3( +0.6336590, +0.0174913, +0.6339004 ),
    float3( +0.8165796, +0.0200828, +0.8168265 ),
    float3( +0.4508830, -0.0892848, +0.4596382 ),
    float3( +0.9695752, -0.1212535, +0.9771277 ),
    float3( +0.5904603, -0.2048051, +0.6249708 ),
    float3( +0.7404402, -0.3184013, +0.8059970 ),
    float3( +0.9107504, -0.3932986, +0.9920434 ),
    float3( +0.2479053, -0.2340817, +0.3409564 ),
    float3( +0.7222927, -0.5845174, +0.9291756 ),
    float3( +0.4767374, -0.4289174, +0.6412867 ),
    float3( +0.4893593, -0.7637584, +0.9070829 ),
    float3( +0.2963522, -0.6137760, +0.6815759 ),
    float3( +0.1755842, -0.4334003, +0.4676170 ),
    float3( +0.1360411, -0.7557332, +0.7678801 ),
    float3( +0.1855755, -0.9548430, +0.9727093 ),
    float3( +0.0002820, -0.5056334, +0.5056335 )
};


static const float3 g_Poisson96[96] =
{
    float3( -0.0403876, -0.8419777, +0.8429458 ),
    float3( -0.0866264, -0.5079851, +0.5153183 ),
    float3( -0.1224081, -0.9850855, +0.9926617 ),
    float3( -0.1226595, -0.6816584, +0.6926063 ),
    float3( -0.1191302, -0.3471802, +0.3670505 ),
    float3( -0.2397694, -0.8340476, +0.8678277 ),
    float3( -0.2812804, -0.6782048, +0.7342209 ),
    float3( -0.2377271, -0.5337023, +0.5842537 ),
    float3( -0.0793476, -0.1517703, +0.1712608 ),
    float3( -0.4919034, -0.8653889, +0.9954230 ),
    float3( -0.4550894, -0.6634924, +0.8045673 ),
    float3( -0.3381177, -0.4022819, +0.5255039 ),
    float3( -0.5085003, -0.5066661, +0.7178322 ),
    float3( -0.6749743, -0.7097090, +0.9794270 ),
    float3( -0.6723632, -0.4928165, +0.8336309 ),
    float3( -0.3238158, -0.1970847, +0.3790766 ),
    float3( -0.5139163, -0.3216180, +0.6062574 ),
    float3( -0.6831340, -0.2914454, +0.7427062 ),
    float3( -0.4764391, -0.1735475, +0.5070631 ),
    float3( -0.8831391, -0.3860794, +0.9638423 ),
    float3( -0.7554776, -0.1553841, +0.7712916 ),
    float3( -0.9237850, -0.1836212, +0.9418574 ),
    float3( -0.5083610, +0.0086067, +0.5084339 ),
    float3( -0.9567527, -0.0078530, +0.9567850 ),
    float3( -0.6818218, +0.0244445, +0.6822599 ),
    float3( -0.2927991, +0.0333949, +0.2946973 ),
    float3( -0.1420011, +0.0395289, +0.1474003 ),
    float3( -0.8947619, +0.1483836, +0.9069822 ),
    float3( -0.7663029, +0.2735212, +0.8136547 ),
    float3( -0.6029718, +0.2360898, +0.6475441 ),
    float3( -0.9012361, +0.3643323, +0.9720929 ),
    float3( -0.4431779, +0.2416853, +0.5047954 ),
    float3( -0.6167140, +0.4098776, +0.7404970 ),
    float3( -0.7698247, +0.5252072, +0.9319188 ),
    float3( -0.4591635, +0.4254926, +0.6259993 ),
    float3( -0.6193955, +0.5780694, +0.8472397 ),
    float3( -0.1571103, +0.2054507, +0.2586381 ),
    float3( -0.4123918, +0.5897211, +0.7196096 ),
    float3( -0.5237168, +0.7524166, +0.9167388 ),
    float3( -0.2315706, +0.4110785, +0.4718162 ),
    float3( -0.4324275, +0.9015638, +0.9999054 ),
    float3( -0.2602250, +0.7798824, +0.8221518 ),
    float3( -0.1855088, +0.6405326, +0.6668550 ),
    float3( -0.0631948, +0.3238317, +0.3299402 ),
    float3( -0.2361725, +0.9591521, +0.9878007 ),
    float3( -0.0018598, +0.1074120, +0.1074281 ),
    float3( -0.0804199, +0.7839980, +0.7881118 ),
    float3( +0.0137250, +0.5012080, +0.5013959 ),
    float3( +0.0302112, +0.6611616, +0.6618515 ),
    float3( +0.0163704, +0.9598445, +0.9599841 ),
    float3( +0.1857906, +0.9584860, +0.9763266 ),
    float3( +0.0784874, +0.2417331, +0.2541558 ),
    float3( +0.1357376, +0.4062127, +0.4282913 ),
    float3( +0.1845639, +0.5740392, +0.6029800 ),
    float3( +0.2254979, +0.7750816, +0.8072179 ),
    float3( +0.3838611, +0.8303300, +0.9147663 ),
    float3( +0.2958074, +0.4314820, +0.5231431 ),
    float3( +0.4304548, +0.6814911, +0.8060530 ),
    float3( +0.5370785, +0.7913437, +0.9563881 ),
    float3( +0.4443785, +0.5258204, +0.6884470 ),
    float3( +0.5771415, +0.6401811, +0.8619305 ),
    float3( +0.3623219, +0.2960911, +0.4679179 ),
    float3( +0.7255664, +0.6867011, +0.9990020 ),
    float3( +0.6815006, +0.5108145, +0.8516892 ),
    float3( +0.8464920, +0.5122826, +0.9894353 ),
    float3( +0.6020624, +0.2977475, +0.6716641 ),
    float3( +0.8042987, +0.3536090, +0.8785987 ),
    float3( +0.2394170, +0.0792043, +0.2521782 ),
    float3( +0.4519147, +0.1219826, +0.4680883 ),
    float3( +0.9526030, +0.2988966, +0.9983945 ),
    float3( +0.7082511, +0.1612283, +0.7263706 ),
    float3( +0.8462632, +0.0930516, +0.8513636 ),
    float3( +0.6101166, +0.0365563, +0.6112108 ),
    float3( +0.9863577, -0.1182441, +0.9934199 ),
    float3( +0.8190978, -0.1294892, +0.8292699 ),
    float3( +0.6563655, -0.1232929, +0.6678450 ),
    float3( +0.2826931, -0.1012181, +0.3002674 ),
    float3( +0.4911776, -0.1628683, +0.5174761 ),
    float3( +0.1163677, -0.0484713, +0.1260591 ),
    float3( +0.8974063, -0.2732542, +0.9380863 ),
    float3( +0.7553440, -0.3278418, +0.8234226 ),
    float3( +0.5750262, -0.3089627, +0.6527734 ),
    float3( +0.8830774, -0.4400037, +0.9866250 ),
    float3( +0.3707938, -0.2564998, +0.4508660 ),
    float3( +0.6983998, -0.5076644, +0.8634149 ),
    float3( +0.4854268, -0.4372651, +0.6533299 ),
    float3( +0.7143911, -0.6611294, +0.9733688 ),
    float3( +0.1537492, -0.2076075, +0.2583402 ),
    float3( +0.2936103, -0.3914900, +0.4893582 ),
    float3( +0.5420131, -0.7008795, +0.8860081 ),
    float3( +0.2966139, -0.6110919, +0.6792740 ),
    float3( +0.3792824, -0.8804409, +0.9586612 ),
    float3( +0.1106483, -0.3808194, +0.3965684 ),
    float3( +0.2513747, -0.7631646, +0.8034982 ),
    float3( +0.1178393, -0.6159950, +0.6271650 ),
    float3( +0.1382435, -0.9177985, +0.9281516 )
};


static const float3 g_Poisson128[128] =
{
    float3( -0.7089940, -0.6214720, +0.9428149 ),
    float3( -0.5671330, -0.6822230, +0.8871686 ),
    float3( -0.9786960, -0.1986800, +0.9986589 ),
    float3( -0.8081850, -0.4610650, +0.9304536 ),
    float3( -0.9891760, +0.1190640, +0.9963159 ),
    float3( -0.9409420, +0.2577720, +0.9756117 ),
    float3( -0.5741980, +0.7546090, +0.9482289 ),
    float3( -0.7324710, +0.6592870, +0.9854811 ),
    float3( -0.0525410, -0.8784000, +0.8799700 ),
    float3( -0.1610590, -0.9578300, +0.9712766 ),
    float3( -0.3792030, -0.4380960, +0.5794161 ),
    float3( -0.3822610, -0.3019270, +0.4871174 ),
    float3( -0.3764430, +0.4197840, +0.5638510 ),
    float3( -0.4956710, +0.3263450, +0.5934566 ),
    float3( -0.4039250, +0.8622360, +0.9521587 ),
    float3( -0.1245920, +0.9874620, +0.9952911 ),
    float3( +0.0768400, -0.9224200, +0.9256150 ),
    float3( +0.2310630, -0.9303730, +0.9586366 ),
    float3( +0.0274670, -0.0147730, +0.0311878 ),
    float3( +0.2863450, -0.0145380, +0.2867138 ),
    float3( +0.1018970, +0.3734240, +0.3870768 ),
    float3( +0.1954070, +0.4803110, +0.5185389 ),
    float3( +0.0837160, +0.8588590, +0.8629294 ),
    float3( +0.0853160, +0.5643070, +0.5707199 ),
    float3( +0.5281740, -0.7293590, +0.9005178 ),
    float3( +0.6464420, -0.6162280, +0.8930981 ),
    float3( +0.8951860, -0.4160920, +0.9871629 ),
    float3( +0.9366010, -0.2850270, +0.9790106 ),
    float3( +0.6702940, +0.4672220, +0.8170621 ),
    float3( +0.5202950, +0.3433550, +0.6233776 ),
    float3( +0.8388290, +0.5418810, +0.9986336 ),
    float3( +0.5227520, +0.7146860, +0.8854635 ),
    float3( -0.6728040, -0.4958620, +0.8357896 ),
    float3( -0.5258120, -0.4882710, +0.7175561 ),
    float3( -0.6513380, +0.1081620, +0.6602576 ),
    float3( -0.8251040, +0.3527810, +0.8973578 ),
    float3( -0.7794940, +0.5090390, +0.9309842 ),
    float3( -0.5991140, +0.5553780, +0.8169347 ),
    float3( -0.1685700, -0.8036030, +0.8210930 ),
    float3( -0.3734380, -0.9071870, +0.9810424 ),
    float3( -0.1481460, -0.1147090, +0.1873643 ),
    float3( -0.2116510, -0.2408530, +0.3206342 ),
    float3( -0.0487150, +0.4884940, +0.4909170 ),
    float3( -0.3556680, +0.2905100, +0.4592339 ),
    float3( -0.2024920, +0.8664630, +0.8898096 ),
    float3( -0.0534200, +0.8531320, +0.8548028 ),
    float3( +0.2118690, -0.7915620, +0.8194259 ),
    float3( +0.0788500, -0.7116190, +0.7159742 ),
    float3( +0.0428120, -0.3753060, +0.3777399 ),
    float3( +0.1605940, -0.4388590, +0.4673196 ),
    float3( +0.2080920, +0.2891210, +0.3562208 ),
    float3( +0.1810880, +0.1535040, +0.2373949 ),
    float3( +0.4325840, +0.8918800, +0.9912511 ),
    float3( +0.1999780, +0.9297440, +0.9510074 ),
    float3( +0.8003100, -0.5964810, +0.9981412 ),
    float3( +0.5758440, -0.5004610, +0.7629269 ),
    float3( +0.9621100, -0.1146000, +0.9689111 ),
    float3( +0.8309570, -0.1460440, +0.8436933 ),
    float3( +0.5528250, +0.2090580, +0.5910336 ),
    float3( +0.8132120, +0.3897220, +0.9017743 ),
    float3( +0.6562940, +0.7456790, +0.9933574 ),
    float3( +0.5248890, +0.5826980, +0.7842483 ),
    float3( -0.9209760, -0.3289270, +0.9779518 ),
    float3( -0.5581920, -0.3306940, +0.6487964 ),
    float3( -0.8551130, +0.1380860, +0.8661906 ),
    float3( -0.7648520, +0.0022930, +0.7648554 ),
    float3( -0.4507560, -0.7735720, +0.8953182 ),
    float3( -0.3194260, -0.7572700, +0.8218825 ),
    float3( -0.2036160, -0.3950070, +0.4443985 ),
    float3( -0.3067120, -0.1341980, +0.3347856 ),
    float3( -0.2098050, +0.2085110, +0.2957955 ),
    float3( -0.1692030, +0.3812370, +0.4170987 ),
    float3( -0.4147870, +0.7063550, +0.8191371 ),
    float3( -0.2136750, +0.6780910, +0.7109602 ),
    float3( +0.1286520, -0.5657470, +0.5801905 ),
    float3( +0.2864410, -0.6815780, +0.7393220 ),
    float3( +0.2617770, -0.2838630, +0.3861417 ),
    float3( +0.0149550, -0.1624500, +0.1631369 ),
    float3( +0.3157310, +0.1210170, +0.3381289 ),
    float3( +0.4195410, +0.2304200, +0.4786523 ),
    float3( +0.0156850, +0.9764970, +0.9766230 ),
    float3( +0.0651120, +0.7204850, +0.7234212 ),
    float3( +0.6369040, -0.0393900, +0.6381209 ),
    float3( +0.7926540, -0.2760840, +0.8393585 ),
    float3( +0.6903180, +0.3282840, +0.7644013 ),
    float3( +0.7243520, +0.1553030, +0.7408136 ),
    float3( +0.7288860, +0.6330180, +0.9653945 ),
    float3( -0.7309940, -0.3456690, +0.8086033 ),
    float3( -0.9700470, -0.0282520, +0.9704583 ),
    float3( -0.6815790, +0.2408420, +0.7228795 ),
    float3( -0.6794800, +0.4021560, +0.7895711 ),
    float3( -0.4395440, -0.6184920, +0.7587696 ),
    float3( -0.1461510, -0.6460370, +0.6623624 ),
    float3( -0.3946660, -0.0280020, +0.3956581 ),
    float3( -0.0554840, -0.2825210, +0.2879177 ),
    float3( -0.2190420, +0.0017410, +0.2190489 ),
    float3( -0.3102390, +0.1215030, +0.3331835 ),
    float3( -0.0824540, +0.6779430, +0.6829388 ),
    float3( -0.4505440, +0.5801740, +0.7345691 ),
    float3( +0.3955200, -0.8231480, +0.9132408 ),
    float3( +0.4313840, -0.6332400, +0.7662147 ),
    float3( +0.1415130, -0.2275940, +0.2680018 ),
    float3( +0.2887790, -0.1454870, +0.3233570 ),
    float3( +0.3987600, +0.3971440, +0.5627903 ),
    float3( +0.1581700, +0.0172060, +0.1591031 ),
    float3( +0.2391310, +0.6859850, +0.7264703 ),
    float3( +0.3127890, +0.8389260, +0.8953401 ),
    float3( +0.6418050, -0.2599680, +0.6924573 ),
    float3( +0.7577960, -0.4595200, +0.8862355 ),
    float3( +0.8806330, +0.2204820, +0.9078143 ),
    float3( +0.9603210, +0.0590630, +0.9621356 ),
    float3( -0.8280720, -0.2189210, +0.8565218 ),
    float3( -0.6851090, -0.1357820, +0.6984348 ),
    float3( -0.5356260, +0.0065290, +0.5356658 ),
    float3( -0.5174520, +0.1682760, +0.5441263 ),
    float3( -0.2839380, -0.5403500, +0.6104088 ),
    float3( -0.0438950, -0.5046060, +0.5065116 ),
    float3( -0.0533950, +0.1513330, +0.1604765 ),
    float3( -0.0230270, +0.2816990, +0.2826386 ),
    float3( -0.2871450, +0.5419330, +0.6133054 ),
    float3( +0.4847900, -0.1383370, +0.5041413 ),
    float3( +0.3547590, -0.4612630, +0.5819085 ),
    float3( +0.4383540, +0.0239030, +0.4390052 ),
    float3( +0.3910370, +0.6516810, +0.7599987 ),
    float3( +0.3202440, +0.5387190, +0.6267172 ),
    float3( +0.8158000, +0.0179200, +0.8159968 ),
    float3( -0.5388640, -0.1738260, +0.5662066 ),
    float3( +0.4655580, -0.2966590, +0.5520424 )
};
#line 12 "/Plugin/NRD/Private/Reblur/Common.ush"
#line 102 "/Plugin/NRD/Private/Reblur/Common.ush"
float PixelRadiusToWorld( float unproject, float orthoMode, float pixelRadius, float viewZ )
{
     return pixelRadius * unproject * lerp( viewZ, 1.0, abs( orthoMode ) );
}

float GetHitDistFactor( float hitDist, float frustumHeight, float scale = 1.0 )
{
    return saturate( hitDist / ( hitDist * scale + frustumHeight ) );
}

float4 GetBlurKernelRotation(   const uint mode, uint2 pixelPos, float4 baseRotator, uint frameIndex )
{
    if( mode ==  1  )
    {
        float angle = STL::Sequence::Bayer4x4( pixelPos, frameIndex );
        float4 rotator = STL::Geometry::GetRotator( angle * STL::Math::Pi( 2.0 ) );

        baseRotator = STL::Geometry::CombineRotators( baseRotator, rotator );
    }
    else if( mode ==  2  )
    {
        STL::Rng::Initialize( pixelPos, frameIndex );

        float2 rnd = STL::Rng::GetFloat2( );
        float4 rotator = STL::Geometry::GetRotator( rnd.x * STL::Math::Pi( 2.0 ) );
        rotator *= 1.0 + ( rnd.y * 2.0 - 1.0 ) * 0.5;

        baseRotator = STL::Geometry::CombineRotators( baseRotator, rotator );
    }

    return baseRotator;
}

float IsInScreen( float2 uv )
{
    return float( all( saturate( uv ) == uv ) );
}

float2 ApplyCheckerboard( inout float2 uv, uint mode, uint counter, float2 screenSize, float2 invScreenSize, uint frameIndex )
{
    int2 uvi = int2( uv * screenSize );
    bool hasData = STL::Sequence::CheckerBoard( uvi, frameIndex ) == mode;
    if( !hasData )
        uvi.x += ( ( counter & 0x1 ) == 0 ) ? -1 : 1;
    uv = ( float2( uvi ) + 0.5 ) * invScreenSize;

    return float2( uv.x * 0.5, uv.y );
}

float GetSpecMagicCurve( float roughness, float power = 0.25 )
{


    float f = 1.0 - exp2( -200.0 * roughness * roughness );
    f *= STL::Math::Pow01( roughness, power );

    return f;
}

float GetSpecMagicCurve2( float roughness, float percentOfVolume = 0.987 )
{



    float angle = STL::ImportanceSampling::GetSpecularLobeHalfAngle( roughness, percentOfVolume );
    float almostHalfPi = STL::ImportanceSampling::GetSpecularLobeHalfAngle( 1.0, percentOfVolume );

    return saturate( angle / almostHalfPi );
}

float ComputeParallax( float3 X, float2 uvForZeroParallax, float4x4 mWorldToClip, float2 rectSize, float unproject, float orthoMode )
{
    float3 clip = STL::Geometry::ProjectiveTransform( mWorldToClip, X ).xyw;
    clip.xy /= clip.z;
    clip.y = -clip.y;

    float2 uv = clip.xy * 0.5 + 0.5;
    float invDist = orthoMode == 0.0 ? rsqrt( STL::Math::LengthSquared( X ) ) : rcp( clip.z );

    float2 parallaxInUv = uv - uvForZeroParallax;
    float parallaxInPixels = length( parallaxInUv * rectSize );
    float parallaxInUnits = PixelRadiusToWorld( unproject, orthoMode, parallaxInPixels, clip.z );
    float parallax = parallaxInUnits * invDist;

    return parallax *  30.0 ;
}

float GetParallaxInPixels( float parallax, float unproject )
{
    float smbParallaxInPixels = parallax / (  30.0  * unproject );

    return smbParallaxInPixels;
}

float GetColorCompressionExposureForSpatialPasses( float roughness )
{
#line 213 "/Plugin/NRD/Private/Reblur/Common.ush"
        return 0.5 * ( 1.0 - roughness ) / ( 1.0 + 1000.0 * roughness * roughness ) + ( 1.0 - sqrt( saturate( roughness ) ) ) * 0.03;
#line 221 "/Plugin/NRD/Private/Reblur/Common.ush"
}



float EstimateCurvature( float3 Ni, float3 Vi, float3 N, float3 X )
{


    float NoV = dot( Vi, N );
    float3 Xi = 0 + Vi * dot( X - 0, N ) / NoV;
    float3 edge = Xi - X;
    float edgeLenSq = STL::Math::LengthSquared( edge );
    float curvature = dot( Ni - N, edge ) / edgeLenSq;

    return curvature;
}

float ApplyThinLensEquation( float NoV, float hitDist, float curvature )
{
#line 254 "/Plugin/NRD/Private/Reblur/Common.ush"
    float hitDistFocused = 0.5 * hitDist / ( 0.5 + curvature * NoV * hitDist );

    return hitDistFocused;
}

float3 GetXvirtual(
    float NoV, float hitDist, float curvature,
    float3 X, float3 Xprev, float3 V,
    float dominantFactor )
{
#line 293 "/Plugin/NRD/Private/Reblur/Common.ush"
    float hitDistFocused = ApplyThinLensEquation( NoV, hitDist, curvature );


    float compressionRatio = saturate( ( abs( hitDistFocused ) + 1e-6 ) / ( hitDist + 1e-6 ) );

    float3 Xvirtual = lerp( Xprev, X, compressionRatio * dominantFactor ) - V * hitDistFocused * dominantFactor;

    return Xvirtual;
}



float2x3 GetKernelBasis( float3 D, float3 N, float NoD, float worldRadius, float roughness = 1.0, float anisoFade = 1.0 )
{
    float3x3 basis = STL::Geometry::GetBasis( N );

    float3 T = basis[ 0 ];
    float3 B = basis[ 1 ];

    if( roughness < 0.95 && NoD < 0.999 )
    {
        float3 R = reflect( -D, N );
        T = normalize( cross( N, R ) );
        B = cross( R, T );

        float skewFactor = lerp( roughness, 1.0, NoD );
        T *= lerp( skewFactor, 1.0, anisoFade );
    }

    T *= worldRadius;
    B *= worldRadius;

    return float2x3( T, B );
}

float2 GetKernelSampleCoordinates( float4x4 mToClip, float3 offset, float3 X, float3 T, float3 B, float4 rotator = float4( 1, 0, 0, 1 ) )
{
#line 335 "/Plugin/NRD/Private/Reblur/Common.ush"
    offset.xy = STL::Geometry::RotateVector( rotator, offset.xy );

    float3 p = X + T * offset.x + B * offset.y;
    float3 clip = STL::Geometry::ProjectiveTransform( mToClip, p ).xyw;
    clip.xy /= clip.z;
    clip.y = -clip.y;

    float2 uv = clip.xy * 0.5 + 0.5;

    return uv;
}



float2 GetGeometryWeightParams( float planeDistSensitivity, float frustumHeight, float3 Xv, float3 Nv, float scale = 1.0 )
{
    float a = scale / ( planeDistSensitivity * frustumHeight + 1e-6 );
    float b = -dot( Nv, Xv ) * a;

    return float2( a, b );
}

float2 GetHitDistanceWeightParams( float hitDist, float nonLinearAccumSpeed, float roughness = 1.0 )
{
    float smc = GetSpecMagicCurve2( roughness );
    float norm = min( nonLinearAccumSpeed, smc * 0.97 + 0.03 );
    float a = 1.0 / norm;
    float b = hitDist * a;

    return float2( a, -b );
}
#line 394 "/Plugin/NRD/Private/Reblur/Common.ush"
float GetRoughnessWeight( float2 params, float roughness )
{
    return  STL::Math::SmoothStep01( 1.0 - abs( ( roughness ) * ( params.x ) + ( params.y ) ) ) ;
}

float GetHitDistanceWeight( float2 params, float hitDist )
{
    return  rcp( ( - 3.0 * abs( ( hitDist ) * ( params.x ) + ( params.y ) ) ) * ( - 3.0 * abs( ( hitDist ) * ( params.x ) + ( params.y ) ) ) - ( - 3.0 * abs( ( hitDist ) * ( params.x ) + ( params.y ) ) ) + 1.0 ) ;
}

float GetGeometryWeight( float2 params, float3 n0, float3 p )
{
    float d = dot( n0, p );

    return  STL::Math::SmoothStep01( 1.0 - abs( ( d ) * ( params.x ) + ( params.y ) ) ) ;
}

float GetNormalWeight( float param, float3 N, float3 n )
{
    float cosa = saturate( dot( N, n ) );
    float angle = STL::Math::AcosApprox( cosa );

    return  STL::Math::SmoothStep01( 1.0 - abs( ( angle ) * ( param ) + ( 0.0 ) ) ) ;
}

float GetGaussianWeight( float r )
{
    return exp( -0.66 * r * r );
}
#line 23 "/Plugin/NRD/Private/Reblur/REBLUR_DiffuseOcclusion_TemporalAccumulation.cs.usf"
#line 1 "REBLUR_Common.ush"
#line 30 "/Plugin/NRD/Private/Reblur/REBLUR_Common.ush"
float4 PackNormalRoughness( float4 p )
{
    return float4( p.xyz * 0.5 + 0.5, p.w );
}

float4 UnpackNormalAndRoughness( float4 p, bool isNormalized = true )
{
    p.xyz = p.xyz * 2.0 - 1.0;

    if( isNormalized )
        p.xyz = normalize( p.xyz );

    return p;
}

uint PackAccumSpeedsMaterialID( float diffAccumSpeed, float specAccumSpeed, float materialID )
{
    float3 t = float3( diffAccumSpeed, specAccumSpeed, materialID );
    t.xy /=  63.0 ;

    uint p = STL::Packing::RgbaToUint( t.xyzz,  7 ,  7 ,  ( 16 - 7 - 7 ) , 0 );

    return p;
}

float3 UnpackAccumSpeedsMaterialID( uint p )
{
    float3 t = STL::Packing::UintToRgba( p,  7 ,  7 ,  ( 16 - 7 - 7 ) , 0 ).xyz;
    t.xy *=  63.0 ;

    return t;
}



float4 PackInternalData1( float diffAccumSpeed, float diffRadiusScale, float specAccumSpeed, float specRadiusScale )
{
    float4 r;
    r.x = saturate( diffAccumSpeed /  63.0  );
    r.y = diffRadiusScale;
    r.z = saturate( specAccumSpeed /  63.0  );
    r.w = specRadiusScale;


    r.yw = STL::Math::Sqrt01( r.yw );
#line 81 "/Plugin/NRD/Private/Reblur/REBLUR_Common.ush"
    return r;
}

float4 UnpackInternalData1( float4 p )
{
#line 91 "/Plugin/NRD/Private/Reblur/REBLUR_Common.ush"
    float4 r;
    r.x = p.x *  63.0 ;
    r.y = p.y;
    r.z = p.z *  63.0 ;
    r.w = p.w;


    r.yw *= r.yw;

    return r;
}

float4 PackInternalData2( float fbits, float curvature, float virtualHistoryAmount, float hitDistScaleForTracking, float viewZ )
{







    float pixelSize = PixelRadiusToWorld( gUnproject, gOrthoMode, 1.0, viewZ );
    float packedCurvature = curvature * pixelSize;

    fbits += 128.0 * ( packedCurvature < 0.0 ? 1 : 0 );

    float4 r;
    r.x = saturate( ( fbits + 0.5 ) / 255.0 );
    r.y = abs( packedCurvature );
    r.z = virtualHistoryAmount;
    r.w = hitDistScaleForTracking;


    r.yzw = STL::Math::Sqrt01( r.yzw );

    return r;
}

float3 UnpackInternalData2( float4 p, float viewZ, out uint bits )
{

    p.yzw *= p.yzw;

    bits = uint( p.x * 255.0 + 0.5 );

    float pixelSize = PixelRadiusToWorld( gUnproject, gOrthoMode, 1.0, viewZ );
    float sgn = ( bits & 128 ) != 0 ? -1.0 : 1.0;
    float curvature = p.y * sgn / pixelSize;

    return float3( p.zw, curvature );
}



float SaturateParallax( float parallax )
{



    return saturate( 1.0 - exp2( -3.5 * parallax * parallax ) );
}

float GetSpecAccumulatedFrameNum( float roughness, float powerScale )
{


    return  63.0  * GetSpecMagicCurve( roughness,  ( 0.4 + 0.2 * exp2( -gFramerateScale ) )  * powerScale );
}

float AdvanceAccumSpeed( float4 prevAccumSpeed, float4 weights )
{
    float4 accumSpeeds = prevAccumSpeed + 1.0;
    float accumSpeed = STL::Filtering::ApplyBilinearCustomWeights( accumSpeeds.x, accumSpeeds.y, accumSpeeds.z, accumSpeeds.w, weights );

    return min( accumSpeed, gMaxAccumulatedFrameNum );
}

float GetSpecAccumSpeed( float maxAccumSpeed, float roughness, float NoV, float parallax, float curvature, float viewZ )
{

    float smbParallaxNorm = SaturateParallax( parallax *  ( 2.0 * gFramerateScale )  );
    roughness = roughness + saturate( 0.05 - roughness ) * ( 1.0 - smbParallaxNorm );


    float pixelSize = PixelRadiusToWorld( gUnproject, gOrthoMode, 1.0, viewZ );
    float curvatureAngleTan = abs( curvature ) * pixelSize * gFramerateScale;

    float percentOfVolume = 0.75;
    float roughnessFromCurvatureAngle = STL::Math::Sqrt01( curvatureAngleTan * ( 1.0 - percentOfVolume ) / percentOfVolume );

    roughness = lerp( roughness, 1.0, roughnessFromCurvatureAngle );


    float acos01sq = saturate( 1.0 - NoV * 0.99999 );
    float a = STL::Math::Pow01( acos01sq,  ( 1.0 - exp2( -gFramerateScale ) )  );
    float b = 1.1 + roughness * roughness;
    float parallaxSensitivity = ( b + a ) / ( b - a );

    float powerScale = 1.0 + parallaxSensitivity * parallax *  ( 2.0 * gFramerateScale ) ;
    float accumSpeed = GetSpecAccumulatedFrameNum( roughness, powerScale );

    accumSpeed = min( accumSpeed, maxAccumSpeed );

    return accumSpeed * float( gResetHistory == 0 );
}



float3 GetViewVector( float3 X, bool isViewSpace = false )
{
    return gOrthoMode == 0.0 ? normalize( -X ) : ( isViewSpace ? float3( 0, 0, -1 ) : gViewVectorWorld.xyz );
}

float3 GetViewVectorPrev( float3 Xprev, float3 cameraDelta )
{
    return gOrthoMode == 0.0 ? normalize( cameraDelta - Xprev ) : gViewVectorWorldPrev.xyz;
}

float GetMinAllowedLimitForHitDistNonLinearAccumSpeed( float roughness )
{
#line 219 "/Plugin/NRD/Private/Reblur/REBLUR_Common.ush"
    float acceleration = 0.5 * GetSpecMagicCurve2( roughness );

    return 1.0 / ( 1.0 + acceleration * gMaxAccumulatedFrameNum );
}

float InterpolateAccumSpeeds( float surfaceFrameNum, float virtualFrameNum, float virtualMotionAmount )
{




        return lerp( surfaceFrameNum, virtualFrameNum, virtualMotionAmount * virtualMotionAmount );


    float a = 1.0 / ( 1.0 + surfaceFrameNum );
    float b = 1.0 / ( 1.0 + virtualFrameNum );
    float c = lerp( a, b, virtualMotionAmount );

    return 1.0 / c - 1.0;
}

float GetFadeBasedOnAccumulatedFrames( float accumSpeed )
{
    float fade = 1.0;


    fade *= STL::Math::LinearStep(  3.0  - 1,  3.0  + 1, accumSpeed );

    return fade;
}







    float MixHistoryAndCurrent( float history, float current, float nonLinearAccumSpeed, float roughness = 1.0 )
    {
        float r = lerp( history, current, max( nonLinearAccumSpeed, GetMinAllowedLimitForHitDistNonLinearAccumSpeed( roughness ) ) );

        return r;
    }

    float ExtractHitDist( float input )
    { return input; }


    float GetLuma( float2 input )
    { return input.x; }

    float2 ChangeLuma( float2 input, float newLuma )
    { return float2( newLuma, input.y ); }

    float2 ClampNegativeToZero( float2 input )
    { return float2( max( input.x, 0.0 ), input.y ); }


    float GetLuma( float input )
    { return input; }

    float ChangeLuma( float input, float newLuma )
    { return newLuma; }

    float ClampNegativeToZero( float input )
    { return max( input.x, 0.0 ); }
#line 335 "/Plugin/NRD/Private/Reblur/REBLUR_Common.ush"
float2 GetSensitivityToDarkness( float roughness )
{

    float sensitivityToDarknessScale = lerp( 3.0, 1.0, roughness );




        return gSensitivityToDarkness;

}


    float GetColorErrorForAdaptiveRadiusScale( float curr, float prev, float accumSpeed, float roughness = 1.0 )
#line 352 "/Plugin/NRD/Private/Reblur/REBLUR_Common.ush"
{

    float2 p = float2( 0, prev );
    float2 c = float2( 0, curr );
#line 361 "/Plugin/NRD/Private/Reblur/REBLUR_Common.ush"
    float2 f = abs( c - p ) / ( max( c, p ) + GetSensitivityToDarkness( roughness ) );

    float smc = GetSpecMagicCurve2( roughness );
    float level = lerp( 1.0, 0.15, smc );



    float error = max( f.x, f.y );
    error = STL::Math::SmoothStep( 0.0, level, error );
    error *= GetFadeBasedOnAccumulatedFrames( accumSpeed );
    error *= 1.0 - gReference;

    return error;
}

float ComputeAntilagScale(

        float history, float signal, float m1, float sigma,
#line 382 "/Plugin/NRD/Private/Reblur/REBLUR_Common.ush"
    float4 antilagMinMaxThreshold, float2 antilagSigmaScale, float stabilizationStrength,
    float curvatureMulPixelSize, float2 internalData1, float roughness = 1.0
)
{

    m1 = lerp( m1, signal, abs( curvatureMulPixelSize ) );



    float2 h = float2( 0, history );
    float2 c = float2( 0, m1 );
    float2 s = float2( 0, sigma );
#line 403 "/Plugin/NRD/Private/Reblur/REBLUR_Common.ush"
    float2 delta = abs( h - c ) - s * antilagSigmaScale;
    delta /= max( h, c ) + GetSensitivityToDarkness( roughness );

    delta = STL::Math::SmoothStep( antilagMinMaxThreshold.zw, antilagMinMaxThreshold.xy, delta );

    float antilag = min( delta.x, delta.y );
    antilag = lerp( 1.0, antilag, stabilizationStrength );
    antilag = lerp( 1.0, antilag, GetFadeBasedOnAccumulatedFrames( internalData1.x ) );
    antilag = lerp( antilag, 1.0, saturate( internalData1.y ) );
#line 417 "/Plugin/NRD/Private/Reblur/REBLUR_Common.ush"
    return antilag;
}



float GetResponsiveAccumulationAmount( float roughness )
{
    float amount = 1.0 - ( roughness + 1e-6 ) / ( gResponsiveAccumulationRoughnessThreshold + 1e-6 );

    return STL::Math::SmoothStep01( amount );
}

float GetBlurRadius(
    float radius, float radiusBias, float radiusScale,
    float hitDistFactor, float nonLinearAccumSpeed,
    float roughness = 1.0
)
{

    float r = lerp( gMinConvergedStateBaseRadiusScale, 1.0, nonLinearAccumSpeed );


    hitDistFactor = lerp( hitDistFactor, 1.0, nonLinearAccumSpeed );
    r *= hitDistFactor;




    r = radius * r + max( radiusBias, 2.0 * roughness );
    r *= radiusScale;


    r *= GetSpecMagicCurve2( roughness );


    r *= float( radius != 0 );

    return r;
}

float GetBlurRadiusScaleBasingOnTrimming( float roughness, float3 trimmingParams )
{
    float trimmingFactor = NRD_GetTrimmingFactor( roughness, trimmingParams );
    float maxScale = 1.0 + 4.0 * roughness * roughness;
    float scale = lerp( maxScale, 1.0, trimmingFactor );


    return scale;
}



float GetEncodingAwareRoughnessWeights( float roughnessCurr, float roughnessPrev, float fraction )
{
    float a = rcp( lerp( 0.01, 1.0, saturate( roughnessCurr * fraction ) ) );
    float d = abs( roughnessPrev - roughnessCurr );

    return STL::Math::SmoothStep01( 1.0 - ( d -  ( 1.0 / 255.0 )  ) * a );
}

float GetEncodingAwareNormalWeight( float3 Ncurr, float3 Nprev, float maxAngle )
{
    float a = 1.0 / maxAngle;
    float cosa = saturate( dot( Ncurr, Nprev ) );
    float d = STL::Math::AcosApprox( cosa );

    float w = STL::Math::SmoothStep01( 1.0 - ( d -  STL::Math::DegToRad( 0.05 )  ) * a );
    w = saturate( w / 0.95 );

    return w;
}



float GetNormalWeightParams( float nonLinearAccumSpeed, float fraction, float roughness = 1.0 )
{




    float angle = STL::ImportanceSampling::GetSpecularLobeHalfAngle( roughness );
    angle *= lerp( saturate( fraction ), 1.0, nonLinearAccumSpeed );

    return 1.0 / max( angle,  STL::Math::DegToRad( 0.5 )  );
}

float2 GetRoughnessWeightParams( float roughness, float fraction )
{
    float a = rcp( lerp( 0.01, 1.0, saturate( roughness * fraction ) ) );
    float b = roughness * a;

    return float2( a, -b );
}

float2 GetCoarseRoughnessWeightParams( float roughness )
{
    return float2( 1.0, -roughness );
}

float2 GetTemporalAccumulationParams( float isInScreenMulFootprintQuality, float accumSpeed )
{

    float nonLinearAccumSpeed = accumSpeed / ( 1.0 + accumSpeed );
    float norm1 = gFramerateScale * 30.0 * 0.25 / ( 1.0 + gFramerateScale * 30.0 * 0.25 );
    float normAccumSpeed = saturate( nonLinearAccumSpeed / norm1 );


    float w = normAccumSpeed * normAccumSpeed;
    w *= isInScreenMulFootprintQuality;
    w *= float( gResetHistory == 0 );


    float s = normAccumSpeed;
    s *= 1.0 - gReference;

    return float2( w, 1.0 +  ( 3.0 * gFramerateScale )  * s );
}



float GetCombinedWeight
(
    float2 geometryWeightParams, float3 Nv, float3 Xvs,
    float normalWeightParams, float3 N, float4 Ns,
    float2 roughnessWeightParams = 0
)
{
    float3 a = float3( geometryWeightParams.x, normalWeightParams, roughnessWeightParams.x );
    float3 b = float3( geometryWeightParams.y, 0.0, roughnessWeightParams.y );

    float3 t;
    t.x = dot( Nv, Xvs );
    t.y = STL::Math::AcosApprox( saturate( dot( N, Ns.xyz ) ) );
    t.z = Ns.w;

    float3 w =  STL::Math::SmoothStep01( 1.0 - abs( ( t ) * ( a ) + ( b ) ) ) ;

    return w.x * w.y * w.z;
}


void BicubicFilterNoCornersWithFallbackToBilinearFilterWithCustomWeights(
    float2 samplePos, float2 invTextureSize,
    float4 bilinearCustomWeights, bool useBicubic,
    Texture2D<float4> tex0, out float4 c0 )
{
    float2 centerPos = floor( samplePos - 0.5 ) + 0.5; float2 f = saturate( samplePos - centerPos ); float2 f2 = f * f; float2 f3 = f * f2; float2 w0 = - 0.5 * f3 + 2.0 * 0.5 * f2 - 0.5 * f; float2 w1 = ( 2.0 - 0.5 ) * f3 - ( 3.0 - 0.5 ) * f2 + 1.0; float2 w2 = -( 2.0 - 0.5 ) * f3 + ( 3.0 - 2.0 * 0.5 ) * f2 + 0.5 * f; float2 w3 = 0.5 * f3 - 0.5 * f2; float2 w12 = w1 + w2; float2 tc = w2 / w12; float4 w; w.x = w12.x * w0.y; w.y = w0.x * w12.y; w.z = w12.x * w12.y; w.w = w3.x * w12.y; float w4 = w12.x * w3.y; w = useBicubic ? w : bilinearCustomWeights; w4 = useBicubic ? w4 : 0.0; float sum = dot( w, 1.0 ) + w4; float2 uv0 = centerPos + ( useBicubic ? float2( tc.x, -1.0 ) : float2( 0, 0 ) ); float2 uv1 = centerPos + ( useBicubic ? float2( -1.0, tc.y ) : float2( 1, 0 ) ); float2 uv2 = centerPos + ( useBicubic ? float2( tc.x, tc.y ) : float2( 0, 1 ) ); float2 uv3 = centerPos + ( useBicubic ? float2( 2.0, tc.y ) : float2( 1, 1 ) ); float2 uv4 = centerPos + ( useBicubic ? float2( tc.x, 2.0 ) : f ); ;
    c0 = tex0.SampleLevel( gLinearClamp, uv0 * invTextureSize, 0 ) * w.x; c0 += tex0.SampleLevel( gLinearClamp, uv1 * invTextureSize, 0 ) * w.y; c0 += tex0.SampleLevel( gLinearClamp, uv2 * invTextureSize, 0 ) * w.z; c0 += tex0.SampleLevel( gLinearClamp, uv3 * invTextureSize, 0 ) * w.w; c0 += tex0.SampleLevel( gLinearClamp, uv4 * invTextureSize, 0 ) * w4; c0 = sum < 0.0001 ? 0 : c0 * rcp( sum ); ;
}

void BicubicFilterNoCornersWithFallbackToBilinearFilterWithCustomWeights(
    float2 samplePos, float2 invTextureSize,
    float4 bilinearCustomWeights, bool useBicubic,
    Texture2D<float4> tex0, out float4 c0,
    Texture2D<float4> tex1, out float4 c1 )
{
    float2 centerPos = floor( samplePos - 0.5 ) + 0.5; float2 f = saturate( samplePos - centerPos ); float2 f2 = f * f; float2 f3 = f * f2; float2 w0 = - 0.5 * f3 + 2.0 * 0.5 * f2 - 0.5 * f; float2 w1 = ( 2.0 - 0.5 ) * f3 - ( 3.0 - 0.5 ) * f2 + 1.0; float2 w2 = -( 2.0 - 0.5 ) * f3 + ( 3.0 - 2.0 * 0.5 ) * f2 + 0.5 * f; float2 w3 = 0.5 * f3 - 0.5 * f2; float2 w12 = w1 + w2; float2 tc = w2 / w12; float4 w; w.x = w12.x * w0.y; w.y = w0.x * w12.y; w.z = w12.x * w12.y; w.w = w3.x * w12.y; float w4 = w12.x * w3.y; w = useBicubic ? w : bilinearCustomWeights; w4 = useBicubic ? w4 : 0.0; float sum = dot( w, 1.0 ) + w4; float2 uv0 = centerPos + ( useBicubic ? float2( tc.x, -1.0 ) : float2( 0, 0 ) ); float2 uv1 = centerPos + ( useBicubic ? float2( -1.0, tc.y ) : float2( 1, 0 ) ); float2 uv2 = centerPos + ( useBicubic ? float2( tc.x, tc.y ) : float2( 0, 1 ) ); float2 uv3 = centerPos + ( useBicubic ? float2( 2.0, tc.y ) : float2( 1, 1 ) ); float2 uv4 = centerPos + ( useBicubic ? float2( tc.x, 2.0 ) : f ); ;
    c0 = tex0.SampleLevel( gLinearClamp, uv0 * invTextureSize, 0 ) * w.x; c0 += tex0.SampleLevel( gLinearClamp, uv1 * invTextureSize, 0 ) * w.y; c0 += tex0.SampleLevel( gLinearClamp, uv2 * invTextureSize, 0 ) * w.z; c0 += tex0.SampleLevel( gLinearClamp, uv3 * invTextureSize, 0 ) * w.w; c0 += tex0.SampleLevel( gLinearClamp, uv4 * invTextureSize, 0 ) * w4; c0 = sum < 0.0001 ? 0 : c0 * rcp( sum ); ;
    c1 = tex1.SampleLevel( gNearestClamp, centerPos * invTextureSize, 0 ) * bilinearCustomWeights.x; c1 += tex1.SampleLevel( gNearestClamp, centerPos * invTextureSize, 0, int2( 1, 0 ) ) * bilinearCustomWeights.y; c1 += tex1.SampleLevel( gNearestClamp, centerPos * invTextureSize, 0, int2( 0, 1 ) ) * bilinearCustomWeights.z; c1 += tex1.SampleLevel( gNearestClamp, centerPos * invTextureSize, 0, int2( 1, 1 ) ) * bilinearCustomWeights.w; sum = dot( bilinearCustomWeights, 1.0 ); c1 = sum < 0.0001 ? 0 : c1 * rcp( sum ); ;
}

void BicubicFilterNoCornersWithFallbackToBilinearFilterWithCustomWeights(
    float2 samplePos, float2 invTextureSize,
    float4 bilinearCustomWeights, bool useBicubic,
    Texture2D<float4> tex0, out float4 c0,
    Texture2D<float> tex1, out float c1 )
{
    float2 centerPos = floor( samplePos - 0.5 ) + 0.5; float2 f = saturate( samplePos - centerPos ); float2 f2 = f * f; float2 f3 = f * f2; float2 w0 = - 0.5 * f3 + 2.0 * 0.5 * f2 - 0.5 * f; float2 w1 = ( 2.0 - 0.5 ) * f3 - ( 3.0 - 0.5 ) * f2 + 1.0; float2 w2 = -( 2.0 - 0.5 ) * f3 + ( 3.0 - 2.0 * 0.5 ) * f2 + 0.5 * f; float2 w3 = 0.5 * f3 - 0.5 * f2; float2 w12 = w1 + w2; float2 tc = w2 / w12; float4 w; w.x = w12.x * w0.y; w.y = w0.x * w12.y; w.z = w12.x * w12.y; w.w = w3.x * w12.y; float w4 = w12.x * w3.y; w = useBicubic ? w : bilinearCustomWeights; w4 = useBicubic ? w4 : 0.0; float sum = dot( w, 1.0 ) + w4; float2 uv0 = centerPos + ( useBicubic ? float2( tc.x, -1.0 ) : float2( 0, 0 ) ); float2 uv1 = centerPos + ( useBicubic ? float2( -1.0, tc.y ) : float2( 1, 0 ) ); float2 uv2 = centerPos + ( useBicubic ? float2( tc.x, tc.y ) : float2( 0, 1 ) ); float2 uv3 = centerPos + ( useBicubic ? float2( 2.0, tc.y ) : float2( 1, 1 ) ); float2 uv4 = centerPos + ( useBicubic ? float2( tc.x, 2.0 ) : f ); ;
    c0 = tex0.SampleLevel( gLinearClamp, uv0 * invTextureSize, 0 ) * w.x; c0 += tex0.SampleLevel( gLinearClamp, uv1 * invTextureSize, 0 ) * w.y; c0 += tex0.SampleLevel( gLinearClamp, uv2 * invTextureSize, 0 ) * w.z; c0 += tex0.SampleLevel( gLinearClamp, uv3 * invTextureSize, 0 ) * w.w; c0 += tex0.SampleLevel( gLinearClamp, uv4 * invTextureSize, 0 ) * w4; c0 = sum < 0.0001 ? 0 : c0 * rcp( sum ); ;
    c1 = tex1.SampleLevel( gNearestClamp, centerPos * invTextureSize, 0 ) * bilinearCustomWeights.x; c1 += tex1.SampleLevel( gNearestClamp, centerPos * invTextureSize, 0, int2( 1, 0 ) ) * bilinearCustomWeights.y; c1 += tex1.SampleLevel( gNearestClamp, centerPos * invTextureSize, 0, int2( 0, 1 ) ) * bilinearCustomWeights.z; c1 += tex1.SampleLevel( gNearestClamp, centerPos * invTextureSize, 0, int2( 1, 1 ) ) * bilinearCustomWeights.w; sum = dot( bilinearCustomWeights, 1.0 ); c1 = sum < 0.0001 ? 0 : c1 * rcp( sum ); ;
}

void BicubicFilterNoCornersWithFallbackToBilinearFilterWithCustomWeights(
    float2 samplePos, float2 invTextureSize,
    float4 bilinearCustomWeights, bool useBicubic,
    Texture2D<float4> tex0, out float4 c0,
    Texture2D<float4> tex1, out float4 c1,
    Texture2D<float> tex2, out float c2 )
{
    float2 centerPos = floor( samplePos - 0.5 ) + 0.5; float2 f = saturate( samplePos - centerPos ); float2 f2 = f * f; float2 f3 = f * f2; float2 w0 = - 0.5 * f3 + 2.0 * 0.5 * f2 - 0.5 * f; float2 w1 = ( 2.0 - 0.5 ) * f3 - ( 3.0 - 0.5 ) * f2 + 1.0; float2 w2 = -( 2.0 - 0.5 ) * f3 + ( 3.0 - 2.0 * 0.5 ) * f2 + 0.5 * f; float2 w3 = 0.5 * f3 - 0.5 * f2; float2 w12 = w1 + w2; float2 tc = w2 / w12; float4 w; w.x = w12.x * w0.y; w.y = w0.x * w12.y; w.z = w12.x * w12.y; w.w = w3.x * w12.y; float w4 = w12.x * w3.y; w = useBicubic ? w : bilinearCustomWeights; w4 = useBicubic ? w4 : 0.0; float sum = dot( w, 1.0 ) + w4; float2 uv0 = centerPos + ( useBicubic ? float2( tc.x, -1.0 ) : float2( 0, 0 ) ); float2 uv1 = centerPos + ( useBicubic ? float2( -1.0, tc.y ) : float2( 1, 0 ) ); float2 uv2 = centerPos + ( useBicubic ? float2( tc.x, tc.y ) : float2( 0, 1 ) ); float2 uv3 = centerPos + ( useBicubic ? float2( 2.0, tc.y ) : float2( 1, 1 ) ); float2 uv4 = centerPos + ( useBicubic ? float2( tc.x, 2.0 ) : f ); ;
    c0 = tex0.SampleLevel( gLinearClamp, uv0 * invTextureSize, 0 ) * w.x; c0 += tex0.SampleLevel( gLinearClamp, uv1 * invTextureSize, 0 ) * w.y; c0 += tex0.SampleLevel( gLinearClamp, uv2 * invTextureSize, 0 ) * w.z; c0 += tex0.SampleLevel( gLinearClamp, uv3 * invTextureSize, 0 ) * w.w; c0 += tex0.SampleLevel( gLinearClamp, uv4 * invTextureSize, 0 ) * w4; c0 = sum < 0.0001 ? 0 : c0 * rcp( sum ); ;
    c1 = tex1.SampleLevel( gNearestClamp, centerPos * invTextureSize, 0 ) * bilinearCustomWeights.x; c1 += tex1.SampleLevel( gNearestClamp, centerPos * invTextureSize, 0, int2( 1, 0 ) ) * bilinearCustomWeights.y; c1 += tex1.SampleLevel( gNearestClamp, centerPos * invTextureSize, 0, int2( 0, 1 ) ) * bilinearCustomWeights.z; c1 += tex1.SampleLevel( gNearestClamp, centerPos * invTextureSize, 0, int2( 1, 1 ) ) * bilinearCustomWeights.w; sum = dot( bilinearCustomWeights, 1.0 ); c1 = sum < 0.0001 ? 0 : c1 * rcp( sum ); ;
    c2 = tex2.SampleLevel( gNearestClamp, centerPos * invTextureSize, 0 ) * bilinearCustomWeights.x; c2 += tex2.SampleLevel( gNearestClamp, centerPos * invTextureSize, 0, int2( 1, 0 ) ) * bilinearCustomWeights.y; c2 += tex2.SampleLevel( gNearestClamp, centerPos * invTextureSize, 0, int2( 0, 1 ) ) * bilinearCustomWeights.z; c2 += tex2.SampleLevel( gNearestClamp, centerPos * invTextureSize, 0, int2( 1, 1 ) ) * bilinearCustomWeights.w; sum = dot( bilinearCustomWeights, 1.0 ); c2 = sum < 0.0001 ? 0 : c2 * rcp( sum ); ;
}


void BicubicFilterNoCornersWithFallbackToBilinearFilterWithCustomWeights(
    float2 samplePos, float2 invTextureSize,
    float4 bilinearCustomWeights, bool useBicubic,
    Texture2D<float> tex0, out float c0 )
{
    float2 centerPos = floor( samplePos - 0.5 ) + 0.5; float2 f = saturate( samplePos - centerPos ); float2 f2 = f * f; float2 f3 = f * f2; float2 w0 = - 0.5 * f3 + 2.0 * 0.5 * f2 - 0.5 * f; float2 w1 = ( 2.0 - 0.5 ) * f3 - ( 3.0 - 0.5 ) * f2 + 1.0; float2 w2 = -( 2.0 - 0.5 ) * f3 + ( 3.0 - 2.0 * 0.5 ) * f2 + 0.5 * f; float2 w3 = 0.5 * f3 - 0.5 * f2; float2 w12 = w1 + w2; float2 tc = w2 / w12; float4 w; w.x = w12.x * w0.y; w.y = w0.x * w12.y; w.z = w12.x * w12.y; w.w = w3.x * w12.y; float w4 = w12.x * w3.y; w = useBicubic ? w : bilinearCustomWeights; w4 = useBicubic ? w4 : 0.0; float sum = dot( w, 1.0 ) + w4; float2 uv0 = centerPos + ( useBicubic ? float2( tc.x, -1.0 ) : float2( 0, 0 ) ); float2 uv1 = centerPos + ( useBicubic ? float2( -1.0, tc.y ) : float2( 1, 0 ) ); float2 uv2 = centerPos + ( useBicubic ? float2( tc.x, tc.y ) : float2( 0, 1 ) ); float2 uv3 = centerPos + ( useBicubic ? float2( 2.0, tc.y ) : float2( 1, 1 ) ); float2 uv4 = centerPos + ( useBicubic ? float2( tc.x, 2.0 ) : f ); ;
    c0 = tex0.SampleLevel( gLinearClamp, uv0 * invTextureSize, 0 ) * w.x; c0 += tex0.SampleLevel( gLinearClamp, uv1 * invTextureSize, 0 ) * w.y; c0 += tex0.SampleLevel( gLinearClamp, uv2 * invTextureSize, 0 ) * w.z; c0 += tex0.SampleLevel( gLinearClamp, uv3 * invTextureSize, 0 ) * w.w; c0 += tex0.SampleLevel( gLinearClamp, uv4 * invTextureSize, 0 ) * w4; c0 = sum < 0.0001 ? 0 : c0 * rcp( sum ); ;
}

void BicubicFilterNoCornersWithFallbackToBilinearFilterWithCustomWeights(
    float2 samplePos, float2 invTextureSize,
    float4 bilinearCustomWeights, bool useBicubic,
    Texture2D<float> tex0, out float c0,
    Texture2D<float> tex1, out float c1 )
{
    float2 centerPos = floor( samplePos - 0.5 ) + 0.5; float2 f = saturate( samplePos - centerPos ); float2 f2 = f * f; float2 f3 = f * f2; float2 w0 = - 0.5 * f3 + 2.0 * 0.5 * f2 - 0.5 * f; float2 w1 = ( 2.0 - 0.5 ) * f3 - ( 3.0 - 0.5 ) * f2 + 1.0; float2 w2 = -( 2.0 - 0.5 ) * f3 + ( 3.0 - 2.0 * 0.5 ) * f2 + 0.5 * f; float2 w3 = 0.5 * f3 - 0.5 * f2; float2 w12 = w1 + w2; float2 tc = w2 / w12; float4 w; w.x = w12.x * w0.y; w.y = w0.x * w12.y; w.z = w12.x * w12.y; w.w = w3.x * w12.y; float w4 = w12.x * w3.y; w = useBicubic ? w : bilinearCustomWeights; w4 = useBicubic ? w4 : 0.0; float sum = dot( w, 1.0 ) + w4; float2 uv0 = centerPos + ( useBicubic ? float2( tc.x, -1.0 ) : float2( 0, 0 ) ); float2 uv1 = centerPos + ( useBicubic ? float2( -1.0, tc.y ) : float2( 1, 0 ) ); float2 uv2 = centerPos + ( useBicubic ? float2( tc.x, tc.y ) : float2( 0, 1 ) ); float2 uv3 = centerPos + ( useBicubic ? float2( 2.0, tc.y ) : float2( 1, 1 ) ); float2 uv4 = centerPos + ( useBicubic ? float2( tc.x, 2.0 ) : f ); ;
    c0 = tex0.SampleLevel( gLinearClamp, uv0 * invTextureSize, 0 ) * w.x; c0 += tex0.SampleLevel( gLinearClamp, uv1 * invTextureSize, 0 ) * w.y; c0 += tex0.SampleLevel( gLinearClamp, uv2 * invTextureSize, 0 ) * w.z; c0 += tex0.SampleLevel( gLinearClamp, uv3 * invTextureSize, 0 ) * w.w; c0 += tex0.SampleLevel( gLinearClamp, uv4 * invTextureSize, 0 ) * w4; c0 = sum < 0.0001 ? 0 : c0 * rcp( sum ); ;
    c1 = tex1.SampleLevel( gNearestClamp, centerPos * invTextureSize, 0 ) * bilinearCustomWeights.x; c1 += tex1.SampleLevel( gNearestClamp, centerPos * invTextureSize, 0, int2( 1, 0 ) ) * bilinearCustomWeights.y; c1 += tex1.SampleLevel( gNearestClamp, centerPos * invTextureSize, 0, int2( 0, 1 ) ) * bilinearCustomWeights.z; c1 += tex1.SampleLevel( gNearestClamp, centerPos * invTextureSize, 0, int2( 1, 1 ) ) * bilinearCustomWeights.w; sum = dot( bilinearCustomWeights, 1.0 ); c1 = sum < 0.0001 ? 0 : c1 * rcp( sum ); ;
}
#line 24 "/Plugin/NRD/Private/Reblur/REBLUR_DiffuseOcclusion_TemporalAccumulation.cs.usf"
#line 1 "REBLUR_DiffuseSpecular_TemporalAccumulation.ush"
#line 11 "/Plugin/NRD/Private/Reblur/REBLUR_DiffuseSpecular_TemporalAccumulation.ush"
groupshared float4 s_Normal_MinHitDist[  ( 8 + 1 * 2 )  ][  ( 8 + 1 * 2 )  ];

void Preload( uint2 sharedPos, int2 globalPos )
{
    globalPos = clamp( globalPos, 0, gRectSize - 1.0 );
    uint2 globalIdUser = gRectOrigin + globalPos;

    float4 temp = NRD_FrontEnd_UnpackNormalAndRoughness( gIn_Normal_Roughness[ globalIdUser ] );
#line 36 "/Plugin/NRD/Private/Reblur/REBLUR_DiffuseSpecular_TemporalAccumulation.ush"
    s_Normal_MinHitDist[ sharedPos.y ][ sharedPos.x ] = temp;
}

[numthreads(  8 ,  8 , 1 )]
 void  main ( int2 threadPos : SV_GroupThreadId, int2 pixelPos : SV_DispatchThreadId, uint threadIndex : SV_GroupIndex )
{
    uint2 pixelPosUser = gRectOrigin + pixelPos;
    float2 pixelUv = float2( pixelPos + 0.5 ) * gInvRectSize;

    int2 groupBase = pixelPos - threadPos - 1 ; uint stageNum = ( ( 8 + 1 * 2 ) * ( 8 + 1 * 2 ) + 8 * 8 - 1 ) / ( 8 * 8 ); [unroll] for( uint stage = 0; stage < stageNum; stage++ ) { uint virtualIndex = threadIndex + stage * 8 * 8 ; uint2 newId = uint2( virtualIndex % ( 8 + 1 * 2 ) , virtualIndex / ( 8 + 1 * 2 ) ); if( stage == 0 || virtualIndex < ( 8 + 1 * 2 ) * ( 8 + 1 * 2 ) ) Preload( newId, groupBase + newId ); } GroupMemoryBarrierWithGroupSync( ) ;


    float viewZ = abs( gIn_ViewZ[ pixelPosUser ] );

    [branch]
    if( viewZ > gDenoisingRange )
    {


                gOut_Diff[ pixelPos ] = float2( 0,  min( viewZ * 0.125 , 65504.0 )  );
#line 63 "/Plugin/NRD/Private/Reblur/REBLUR_DiffuseSpecular_TemporalAccumulation.ush"
        return;
    }


    float3 Xv = STL::Geometry::ReconstructViewPosition( pixelUv, gFrustum, viewZ, gOrthoMode );
    float3 X = STL::Geometry::RotateVector( gViewToWorld, Xv );


    int2 smemPos = threadPos +  1 ;
    float4 Navg = s_Normal_MinHitDist[ smemPos.y ][ smemPos.x ];

    [unroll]
    for( int dy = 0; dy <=  1  * 2; dy++ )
    {
        [unroll]
        for( int dx = 0; dx <=  1  * 2; dx++ )
        {
            if( dx ==  1  && dy ==  1  )
                continue;

            int2 pos = threadPos + int2( dx, dy );
            float4 t = s_Normal_MinHitDist[ pos.y ][ pos.x ];
#line 90 "/Plugin/NRD/Private/Reblur/REBLUR_DiffuseSpecular_TemporalAccumulation.ush"
            Navg.xyz += t.xyz;
        }
    }

    Navg.xyz /= 9.0;


    float materialID;
    float4 normalAndRoughness = NRD_FrontEnd_UnpackNormalAndRoughness( gIn_Normal_Roughness[ pixelPosUser ], materialID );
    float3 N = normalAndRoughness.xyz;
    float roughness = normalAndRoughness.w;
#line 107 "/Plugin/NRD/Private/Reblur/REBLUR_DiffuseSpecular_TemporalAccumulation.ush"
    float3 motionVector = gIn_ObjectMotion[ pixelPosUser ] * gMotionVectorScale.xyy;
    float2 smbPixelUv = STL::Geometry::GetPrevUvFromMotion( pixelUv, X, gWorldToClipPrev, motionVector, gIsWorldSpaceMotionEnabled );
    float isInScreen = IsInScreen( smbPixelUv );
    float3 Xprev = X + motionVector * float( gIsWorldSpaceMotionEnabled != 0 );


    float curvature = 0;
#line 160 "/Plugin/NRD/Private/Reblur/REBLUR_DiffuseSpecular_TemporalAccumulation.ush"
    STL::Filtering::CatmullRom smbCatromFilter = STL::Filtering::GetCatmullRomFilter( saturate( smbPixelUv ), gRectSizePrev );
    float2 smbCatromGatherUv = smbCatromFilter.origin * gInvScreenSize;
    float4 smbViewZ0 = gIn_Prev_ViewZ.GatherRed( gNearestClamp, smbCatromGatherUv, float2( 1, 1 ) ).wzxy;
    float4 smbViewZ1 = gIn_Prev_ViewZ.GatherRed( gNearestClamp, smbCatromGatherUv, float2( 3, 1 ) ).wzxy;
    float4 smbViewZ2 = gIn_Prev_ViewZ.GatherRed( gNearestClamp, smbCatromGatherUv, float2( 1, 3 ) ).wzxy;
    float4 smbViewZ3 = gIn_Prev_ViewZ.GatherRed( gNearestClamp, smbCatromGatherUv, float2( 3, 3 ) ).wzxy;

    float3 prevViewZ0 =  ( smbViewZ0.yzw / 0.125 ) ;
    float3 prevViewZ1 =  ( smbViewZ1.xzw / 0.125 ) ;
    float3 prevViewZ2 =  ( smbViewZ2.xyw / 0.125 ) ;
    float3 prevViewZ3 =  ( smbViewZ3.xyz / 0.125 ) ;


    STL::Filtering::Bilinear smbBilinearFilter = STL::Filtering::GetBilinearFilter( saturate( smbPixelUv ), gRectSizePrev );

    float2 smbBilinearGatherUv = ( smbBilinearFilter.origin + 1.0 ) * gInvScreenSize;
    float3 prevNflat = UnpackNormalAndRoughness( gIn_Prev_Normal_Roughness.SampleLevel( gLinearClamp, smbBilinearGatherUv, 0 ) ).xyz;
    prevNflat = STL::Geometry::RotateVector( gWorldPrevToWorld, prevNflat );


    uint4 smbPackedAccumSpeedMaterialID = gIn_Prev_AccumSpeeds_MaterialID.GatherRed( gNearestClamp, smbBilinearGatherUv ).wzxy;

    float3 prevAccumSpeedMaterialID00 = UnpackAccumSpeedsMaterialID( smbPackedAccumSpeedMaterialID.x );
    float3 prevAccumSpeedMaterialID10 = UnpackAccumSpeedsMaterialID( smbPackedAccumSpeedMaterialID.y );
    float3 prevAccumSpeedMaterialID01 = UnpackAccumSpeedsMaterialID( smbPackedAccumSpeedMaterialID.z );
    float3 prevAccumSpeedMaterialID11 = UnpackAccumSpeedsMaterialID( smbPackedAccumSpeedMaterialID.w );

    float4 prevDiffAccumSpeeds = float4( prevAccumSpeedMaterialID00.x, prevAccumSpeedMaterialID10.x, prevAccumSpeedMaterialID01.x, prevAccumSpeedMaterialID11.x );
    float4 prevSpecAccumSpeeds = float4( prevAccumSpeedMaterialID00.y, prevAccumSpeedMaterialID10.y, prevAccumSpeedMaterialID01.y, prevAccumSpeedMaterialID11.y );
    float4 prevMaterialIDs = float4( prevAccumSpeedMaterialID00.z, prevAccumSpeedMaterialID10.z, prevAccumSpeedMaterialID01.z, prevAccumSpeedMaterialID11.z );


    float smbParallax = ComputeParallax( Xprev - gCameraDelta.xyz, gOrthoMode == 0.0 ? pixelUv : smbPixelUv, gWorldToClip, gRectSize, gUnproject, gOrthoMode );
    float smbParallaxInPixels = GetParallaxInPixels( smbParallax, gUnproject );


    float3 V = GetViewVector( X );
    float NoV = abs( dot( N, V ) );
    float frustumHeight = PixelRadiusToWorld( gUnproject, gOrthoMode, gRectSize.y, viewZ );
    float mvLengthFactor = saturate( smbParallaxInPixels / 0.5 );
    float frontFacing = lerp( cos( STL::Math::DegToRad( 135.0 ) ), cos( STL::Math::DegToRad( 91.0 ) ), mvLengthFactor );
    float NavgLen = length( Navg.xyz );
    float disocclusionThreshold = ( isInScreen && dot( prevNflat, Navg.xyz ) > frontFacing * NavgLen ) ? gDisocclusionThreshold : -1.0;
    float3 Xvprev = STL::Geometry::AffineTransform( gWorldToViewPrev, Xprev );
    float NoVmod = NoV / frustumHeight;
    float NoVmodMulXvprevz = Xvprev.z * NoVmod;
    float3 planeDist0 = abs( prevViewZ0 * NoVmod - NoVmodMulXvprevz );
    float3 planeDist1 = abs( prevViewZ1 * NoVmod - NoVmodMulXvprevz );
    float3 planeDist2 = abs( prevViewZ2 * NoVmod - NoVmodMulXvprevz );
    float3 planeDist3 = abs( prevViewZ3 * NoVmod - NoVmodMulXvprevz );
    float3 smbOcclusion0 = step( planeDist0, disocclusionThreshold );
    float3 smbOcclusion1 = step( planeDist1, disocclusionThreshold );
    float3 smbOcclusion2 = step( planeDist2, disocclusionThreshold );
    float3 smbOcclusion3 = step( planeDist3, disocclusionThreshold );

    float4 smbOcclusionWeights = STL::Filtering::GetBilinearCustomWeights( smbBilinearFilter, float4( smbOcclusion0.z, smbOcclusion1.y, smbOcclusion2.y, smbOcclusion3.x ) );
    bool smbIsCatromAllowed = dot( smbOcclusion0 + smbOcclusion1 + smbOcclusion2 + smbOcclusion3, 1.0 ) > 11.5 &&  1 ;

    float footprintQuality = STL::Filtering::ApplyBilinearFilter( smbOcclusion0.z, smbOcclusion1.y, smbOcclusion2.y, smbOcclusion3.x, smbBilinearFilter );
    footprintQuality = STL::Math::Sqrt01( footprintQuality );


    float4 materialCmps =  1.0 ;
    smbOcclusion0.z *= materialCmps.x;
    smbOcclusion1.y *= materialCmps.y;
    smbOcclusion2.y *= materialCmps.z;
    smbOcclusion3.x *= materialCmps.w;

    float4 smbOcclusionWeightsWithMaterialID = STL::Filtering::GetBilinearCustomWeights( smbBilinearFilter, float4( smbOcclusion0.z, smbOcclusion1.y, smbOcclusion2.y, smbOcclusion3.x ) );
    bool smbIsCatromAllowedWithMaterialID = dot( smbOcclusion0 + smbOcclusion1 + smbOcclusion2 + smbOcclusion3, 1.0 ) > 11.5 &&  1 ;

    float footprintQualityWithMaterialID = STL::Filtering::ApplyBilinearFilter( smbOcclusion0.z, smbOcclusion1.y, smbOcclusion2.y, smbOcclusion3.x, smbBilinearFilter );
    footprintQualityWithMaterialID = STL::Math::Sqrt01( footprintQualityWithMaterialID );


    float3 Vprev = GetViewVectorPrev( Xprev, gCameraDelta.xyz );
    float NoVprev = abs( dot( N, Vprev ) );
    float sizeQuality = ( NoVprev + 1e-3 ) / ( NoV + 1e-3 );
    sizeQuality *= sizeQuality;
    sizeQuality = lerp( 0.1, 1.0, saturate( sizeQuality ) );
    footprintQuality *= sizeQuality;
    footprintQualityWithMaterialID *= sizeQuality;


    float fbits = smbIsCatromAllowed * 2.0;
    fbits += smbOcclusion0.z * 4.0 + smbOcclusion1.y * 8.0 + smbOcclusion2.y * 16.0 + smbOcclusion3.x * 32.0;



        float diffAccumSpeed = AdvanceAccumSpeed( prevDiffAccumSpeeds, gDiffMaterialMask ? smbOcclusionWeightsWithMaterialID : smbOcclusionWeights );
        diffAccumSpeed *= lerp( gDiffMaterialMask ? footprintQualityWithMaterialID : footprintQuality, 1.0, 1.0 / ( 1.0 + diffAccumSpeed ) );
#line 268 "/Plugin/NRD/Private/Reblur/REBLUR_DiffuseSpecular_TemporalAccumulation.ush"
    uint checkerboard = STL::Sequence::CheckerBoard( pixelPos, gFrameIndex );

        int3 checkerboardPos = pixelPosUser.xyx + int3( -1, 0, 1 );
        float viewZ0 = gIn_ViewZ[ checkerboardPos.xy ];
        float viewZ1 = gIn_ViewZ[ checkerboardPos.zy ];
        float2 wc =  STL::Math::LinearStep( 0.03 , 0.0, abs( float2( viewZ0, viewZ1 ) - viewZ ) * rcp( max( abs( float2( viewZ0, viewZ1 ) ), abs( viewZ ) ) + 1e-6 ) ) ;
        wc *= STL::Math::PositiveRcp( wc.x + wc.y );




        bool diffHasData = gDiffCheckerboard == 2 || checkerboard == gDiffCheckerboard;

            uint diffShift = gDiffCheckerboard != 2 ? 1 : 0;
            uint2 diffPos = uint2( pixelPos.x >> diffShift, pixelPos.y ) + gRectOrigin;
#line 287 "/Plugin/NRD/Private/Reblur/REBLUR_DiffuseSpecular_TemporalAccumulation.ush"
        float  diff = gIn_Diff[ diffPos ];
#line 293 "/Plugin/NRD/Private/Reblur/REBLUR_DiffuseSpecular_TemporalAccumulation.ush"
        float  smbDiffHistory;
        float4 smbDiffShHistory;
        float smbDiffFastHistory;
        BicubicFilterNoCornersWithFallbackToBilinearFilterWithCustomWeights(
            saturate( smbPixelUv ) * gRectSizePrev, gInvScreenSize,
            gDiffMaterialMask ? smbOcclusionWeightsWithMaterialID : smbOcclusionWeights, gDiffMaterialMask ? smbIsCatromAllowedWithMaterialID : smbIsCatromAllowed,
            gIn_Diff_History, smbDiffHistory
#line 306 "/Plugin/NRD/Private/Reblur/REBLUR_DiffuseSpecular_TemporalAccumulation.ush"
        );


        smbDiffHistory = ClampNegativeToZero( smbDiffHistory );



            float d0 = gIn_Diff[ uint2( ( pixelPos.x - 1 ) >> diffShift, pixelPos.y ) + gRectOrigin ];
            float d1 = gIn_Diff[ uint2( ( pixelPos.x + 1 ) >> diffShift, pixelPos.y ) + gRectOrigin ];

            if( !diffHasData )
            {
                diff *= saturate( 1.0 - wc.x - wc.y );
                diff += d0 * wc.x + d1 * wc.y;
            }


        float diffAccumSpeedNonLinear = 1.0 / ( min( diffAccumSpeed, gMaxAccumulatedFrameNum ) + 1.0 );
        if( !diffHasData )
            diffAccumSpeedNonLinear *= 1.0 - gCheckerboardResolveAccumSpeed * diffAccumSpeed / ( 1.0 + diffAccumSpeed );

        float  diffResult = MixHistoryAndCurrent( smbDiffHistory, diff, diffAccumSpeedNonLinear );
#line 333 "/Plugin/NRD/Private/Reblur/REBLUR_DiffuseSpecular_TemporalAccumulation.ush"
        float diffError = GetColorErrorForAdaptiveRadiusScale( diffResult, smbDiffHistory, diffAccumSpeed );



            gOut_Diff[ pixelPos ] = float2( diffResult,  min( viewZ * 0.125 , 65504.0 )  );
#line 617 "/Plugin/NRD/Private/Reblur/REBLUR_DiffuseSpecular_TemporalAccumulation.ush"
        float specAccumSpeed = 0;
        float specError = 0;
        float virtualHistoryAmount = 0;
        float hitDistScaleForTracking = 0;



    gOut_Data1[ pixelPos ] = PackInternalData1( diffAccumSpeed, diffError, specAccumSpeed, specError );
#line 629 "/Plugin/NRD/Private/Reblur/REBLUR_DiffuseSpecular_TemporalAccumulation.ush"
}
#line 25 "/Plugin/NRD/Private/Reblur/REBLUR_DiffuseOcclusion_TemporalAccumulation.cs.usf"
