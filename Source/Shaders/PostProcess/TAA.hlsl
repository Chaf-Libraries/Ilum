#include "../Common.hlsli"

Texture2D              PrevImage : register(t0);
SamplerState           LinearSampler : register(s0);
Texture2D              InImage : register(t1);
SamplerState           PointSampler : register(s1);
Texture2D              GBuffer1 : register(t2);
Texture2D              GBuffer3 : register(t3);
RWTexture2D<float4>    OutImage : register(u4);
ConstantBuffer<Camera> camera : register(b5);

[[vk::push_constant]]
struct
{
	float feedback_min;
	float feedback_max;
	uint  sharpen;
}push_constants;

struct CSParam
{
	uint3 DispatchThreadID : SV_DispatchThreadID;
};

float3 RGBToYCoCg(float3 RGB)
{
	// Y = R/4 + G/2 + B/4
	// Co = R/2 - B/2
	// Cg = -R/4 + G/2 - B/4
	return float3(
	    RGB.r / 4.0 + RGB.g / 2.0 + RGB.b / 4.0,
	    RGB.r / 2.0 - RGB.b / 2.0,
	    -RGB.r / 4.0 + RGB.g / 2.0 - RGB.b / 4.0);
}

float3 YCoCgToRGB(float3 YCoCg)
{
	// R = Y + Co - Cg
	// G = Y + Cg
	// B = Y - Co - Cg
	return clamp(float3(
	                 YCoCg.r + YCoCg.g - YCoCg.b,
	                 YCoCg.r + YCoCg.b,
	                 YCoCg.r - YCoCg.g - YCoCg.b),
	             0.0,
	             1.0);
}

float3 Tonemap(float3 color)
{
	return color / (color + float3(1.0, 1.0, 1.0));
}

float3 InverseTonemap(float3 color)
{
	return color / max((float3(1.0, 1.0, 1.0) - color), float3(1e-8, 1e-8, 1e-8));
}

float3 FindClosestFragment3x3(float2 uv, float2 texel_size)
{
	float2 du = float2(texel_size.x, 0.0);
	float2 dv = float2(0.0, texel_size.y);

	float3 dtl = float3(-1, -1, GBuffer1.SampleLevel(PointSampler, uv - dv - du, 0.0).a);
	float3 dtc = float3(0, -1, GBuffer1.SampleLevel(PointSampler, uv - dv, 0.0).a);
	float3 dtr = float3(1, -1, GBuffer1.SampleLevel(PointSampler, uv - dv + du, 0.0).a);

	float3 dml = float3(-1, 0, GBuffer1.SampleLevel(PointSampler, uv - du, 0.0).a);
	float3 dmc = float3(0, 0, GBuffer1.SampleLevel(PointSampler, uv, 0.0).a);
	float3 dmr = float3(1, 0, GBuffer1.SampleLevel(PointSampler, uv + du, 0.0).a);

	float3 dbl = float3(-1, 1, GBuffer1.SampleLevel(PointSampler, uv + dv - du, 0.0).a);
	float3 dbc = float3(0, 1, GBuffer1.SampleLevel(PointSampler, uv + dv, 0.0).a);
	float3 dbr = float3(1, 1, GBuffer1.SampleLevel(PointSampler, uv + dv + du, 0.0).a);

	float3 dmin = dtl;
	if (dmin.z > dtc.z)
		dmin = dtc;
	if (dmin.z > dtr.z)
		dmin = dtr;

	if (dmin.z > dml.z)
		dmin = dml;
	if (dmin.z > dmc.z)
		dmin = dmc;
	if (dmin.z > dmr.z)
		dmin = dmr;

	if (dmin.z > dbl.z)
		dmin = dbl;
	if (dmin.z > dbc.z)
		dmin = dbc;
	if (dmin.z > dbr.z)
		dmin = dbr;

	return float3(uv + texel_size.xy * dmin.xy, dmin.z);
}

float3 ClipAABB(float3 aabb_min, float3 aabb_max, float3 prev_sample)
{
	float3 p_clip = 0.5 * (aabb_max + aabb_min);
	float3 e_clip = 0.5 * (aabb_max - aabb_min);

	float3 v_clip  = prev_sample - p_clip;
	float3 v_unit  = v_clip.xyz / e_clip;
	float3 a_unit  = abs(v_unit);
	float  ma_unit = max(a_unit.x, max(a_unit.y, a_unit.z));

	if (ma_unit > 1.0)
	{
		return p_clip + v_clip / ma_unit;
	}
	else
	{
		return prev_sample;
	}
}

float3 TemporalReprojection(float2 uv, float2 velocity, float depth, float2 texel_size)
{
	float3 prev_color    = RGBToYCoCg(PrevImage.SampleLevel(LinearSampler, uv - velocity, 0.0).rgb);
	float3 current_color = RGBToYCoCg(InImage.SampleLevel(PointSampler, uv - camera.jitter, 0.0).rgb);

	float2 du = float2(texel_size.x, 0.0);
	float2 dv = float2(0.0, texel_size.y);

	// Neiborhood
	float3 ctl = RGBToYCoCg(InImage.SampleLevel(PointSampler, uv - du - dv, 0.0).rgb);
	float3 ctc = RGBToYCoCg(InImage.SampleLevel(PointSampler, uv - dv, 0.0).rgb);
	float3 ctr = RGBToYCoCg(InImage.SampleLevel(PointSampler, uv + du - dv, 0.0).rgb);

	float3 cml = RGBToYCoCg(InImage.SampleLevel(PointSampler, uv - du, 0.0).rgb);
	float3 cmc = RGBToYCoCg(InImage.SampleLevel(PointSampler, uv, 0.0).rgb);
	float3 cmr = RGBToYCoCg(InImage.SampleLevel(PointSampler, uv + du, 0.0).rgb);

	float3 cbl = RGBToYCoCg(InImage.SampleLevel(PointSampler, uv - du + dv, 0.0).rgb);
	float3 cbc = RGBToYCoCg(InImage.SampleLevel(PointSampler, uv + dv, 0.0).rgb);
	float3 cbr = RGBToYCoCg(InImage.SampleLevel(PointSampler, uv + du + dv, 0.0).rgb);

	float3 cmin = min(ctl, min(ctc, min(ctr, min(cml, min(cmc, min(cmr, min(cbl, min(cbc, cbr))))))));
	float3 cmax = max(ctl, max(ctc, max(ctr, max(cml, max(cmc, max(cmr, max(cbl, max(cbc, cbr))))))));
	float3 cavg = (ctl + ctc + ctr + cml + cmc + cmr + cbl + cbc + cbr) / 9.0;

	float2 chroma_extent = 0.25 * 0.5 * float2(cmax.r - cmin.r, cmax.r - cmin.r);
	float2 chroma_center = current_color.gb;
	cmin.yz              = chroma_center - chroma_extent;
	cmax.yz              = chroma_center + chroma_extent;
	cavg.yz              = chroma_center;

	prev_color = ClipAABB(cmin.xyz, cmax.xyz, prev_color);

	float current_lum = current_color.r;
	float prev_lum    = prev_color.g;

	float unbiased_diff       = abs(current_lum - prev_lum) / max(current_lum, max(prev_lum, 0.2));
	float unbiased_weight     = 1.0 - unbiased_diff;
	float unbiased_weight_sqr = unbiased_weight * unbiased_weight;
	float k_feedback          = lerp(push_constants.feedback_min, push_constants.feedback_max, unbiased_weight_sqr);

	if (push_constants.sharpen == 1)
	{
		float3 sum = float3(0.0, 0.0, 0.0);

		sum += -1.0 * cml;
		sum += -1.0 * ctc;
		sum += 5.0 * current_color;
		sum += -1.0 * cbc;
		sum += -1.0 * cmc;

		current_color = sum;
	}

	prev_color    = Tonemap(prev_color.rgb);
	current_color = Tonemap(current_color.rgb);

	float3 blended = lerp(current_color, prev_color, k_feedback);

	return YCoCgToRGB(InverseTonemap(blended));
}

[numthreads(32, 32, 1)] void main(CSParam param) {
	uint2 extent;
	OutImage.GetDimensions(extent.x, extent.y);

	if (param.DispatchThreadID.x > extent.x || param.DispatchThreadID.y > extent.y)
	{
		return;
	}

	float2 pixel_size = 1.0 / float2(extent);

	float2 uv = (float2(param.DispatchThreadID.xy + float2(0.5, 0.5))) / float2(extent);

	float3 frag = FindClosestFragment3x3(uv - camera.jitter, pixel_size);

	float  depth    = frag.z;
	float2 velocity = GBuffer3.SampleLevel(LinearSampler, frag.xy, 0.0).ba;

	float3 blend = TemporalReprojection(uv, velocity, depth, pixel_size);

	OutImage[param.DispatchThreadID.xy] = float4(blend, 1.0);
}