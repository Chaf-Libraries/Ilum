#include "../../Common.hlsli"
#include "../../Light.hlsli"

#define LOCAL_SIZE 32

ConstantBuffer<Camera> camera : register(b0);
Texture2D GBuffer0 : register(t2);
Texture2D GBuffer1 : register(t3);
Texture2D GBuffer2 : register(t4);
Texture2D GBuffer3 : register(t5);
Texture2D DepthBuffer : register(t6);
StructuredBuffer<MaterialData> materials : register(t7);
StructuredBuffer<DirectionalLight> directional_lights : register(t8);
StructuredBuffer<PointLight> point_lights : register(t9);
StructuredBuffer<SpotLight> spot_lights : register(t10);
RWTexture2D<float4> Lighting : register(u11);

[[vk::push_constant]]
struct
{
    uint directional_light_count;
    uint spot_light_count;
    uint point_light_count;
    uint enable_multi_bounce;
} push_constants;


struct CSParam
{
    uint3 DispatchThreadID : SV_DispatchThreadID;
};

#define sqr(x) x * x

float3 WorldPositionFromDepth(float2 uv, float depth, float4x4 view_projection_inverse)
{
    uv.y = 1.0 - uv.y;
    float2 screen_pos = uv * 2.0 - 1.0;
    float4 ndc_pos = float4(screen_pos, depth, 1.0);
    float4 world_pos = mul(view_projection_inverse, ndc_pos);
    world_pos = world_pos / world_pos.w;
    return world_pos.xyz;
}

float SchlickFresnel(float u)
{
    float m = clamp(1 - u, 0, 1);
    float m2 = m * m;
    return m2 * m2 * m; // pow(m,5)
}

float GTR1(float NdotH, float a)
{
    if (a >= 1)
    {
        return 1 / PI;
    }
    float a2 = a * a;
    float t = 1 + (a2 - 1) * NdotH * NdotH;
    return (a2 - 1) / (PI * log(a2) * t);
}

float GTR2(float NdotH, float a)
{
    float a2 = a * a;
    float t = 1 + (a2 - 1) * NdotH * NdotH;
    return a2 / (PI * t * t);
}

float GTR2_aniso(float NdotH, float HdotX, float HdotY, float ax, float ay)
{
    return 1 / (PI * ax * ay * sqr(sqr(HdotX / ax) + sqr(HdotY / ay) + NdotH * NdotH));
}

float smithG_GGX(float NdotV, float alphaG)
{
    float a = alphaG * alphaG;
    float b = NdotV * NdotV;
    return 1 / (NdotV + sqrt(a + b - a * b));
}

float smithG_GGX_aniso(float NdotV, float VdotX, float VdotY, float ax, float ay)
{
    return 1 / (NdotV + sqrt(sqr(VdotX * ax) + sqr(VdotY * ay) + sqr(NdotV)));
}

float3 mon2lin(float3 x)
{
    return pow(x, float3(2.2, 2.2, 2.2));
}

float3 MatteBRDF(float3 L, float3 V, float3 N, Material material)
{
    if (material.roughness == 0.0)
    {
        // Use Lambert
        return material.base_color.rgb / PI;
    }
    
    // Use Oren Nayar
    float VdotN = dot(V, N);
    float LdotN = dot(L, N);
    float theta_r = acos(VdotN);
    float sigma2 = pow(material.roughness * PI / 180, 2);

    float cos_phi_diff = dot(normalize(V - N * (VdotN)), normalize(L - N * (LdotN)));
    float theta_i = acos(LdotN);
    float alpha = max(theta_i, theta_r);
    float beta = min(theta_i, theta_r);
    
    if (alpha > PI / 2)
    {
        return float3(0.0, 0.0, 0.0);
    }

    float C1 = 1 - 0.5 * sigma2 / (sigma2 + 0.33);
    float C2 = 0.45 * sigma2 / (sigma2 + 0.09);
    
    if (cos_phi_diff >= 0)
    {
        C2 *= sin(alpha);
    }
    else
    {
        C2 *= (sin(alpha) - pow(2 * beta / PI, 3));
    }
    
    float C3 = 0.125 * sigma2 / (sigma2 + 0.09) * pow((4 * alpha * beta) / (PI * PI), 2);
    float3 L1 = material.base_color.rgb / PI * (C1 + cos_phi_diff * C2 * tan(beta) + (1 - abs(cos_phi_diff)) * C3 * tan((alpha + beta) / 2));
    float3 L2 = 0.17 * material.base_color.rgb * material.base_color.rgb / PI * sigma2 / (sigma2 + 0.13) * (1 - cos_phi_diff * (4 * beta * beta) / (PI * PI));
    
    return L1 + L2;
}

float3 DisneyBRDF(float3 L, float3 V, float3 N, Material material)
{
    const float3 ref = abs(dot(N, float3(0, 1, 0))) > 0.99f ? float3(0, 0, 1) : float3(0, 1, 0);

    float3 X = normalize(cross(ref, N));
    float3 Y = cross(N, X);
    
    float NdotL = dot(N, L);
    float NdotV = dot(N, V);
    if (NdotL < 0 || NdotV < 0)
    {
        return float3(0.0, 0.0, 0.0);
    }

    float3 H = normalize(L + V);
    float NdotH = dot(N, H);
    float LdotH = dot(L, H);

    float3 Cdlin = mon2lin(material.base_color.rgb);
    float Cdlum = dot(Cdlin, float3(0.3, 0.6, 0.1)); // luminance approx.

    float3 Ctint = Cdlum > 0 ? Cdlin / Cdlum : float3(1.0, 1.0, 1.0); // normalize lum. to isolate hue+sat
    float3 Cspec0 = lerp(material.specular * .08 * lerp(float3(1.0, 1.0, 1.0), Ctint, material.specular_tint), Cdlin, material.metallic);
    float3 Csheen = lerp(float3(1.0, 1.0, 1.0), Ctint, material.sheen_tint);

    // Diffuse fresnel - go from 1 at normal incidence to .5 at grazing
    // and lerp in diffuse retro-reflection based on roughness
    float FL = SchlickFresnel(NdotL), FV = SchlickFresnel(NdotV);
    float Fd90 = 0.5 + 2 * LdotH * LdotH * material.roughness;
    float Fd = lerp(1.0, Fd90, FL) * lerp(1.0, Fd90, FV);

    // Based on Hanrahan-Krueger brdf approximation of isotropic bssrdf
    // 1.25 scale is used to (roughly) preserve albedo
    // Fss90 used to "flatten" retroreflection based on roughness
    float Fss90 = LdotH * LdotH * material.roughness;
    float Fss = lerp(1.0, Fss90, FL) * lerp(1.0, Fss90, FV);
    float ss = 1.25 * (Fss * (1 / (NdotL + NdotV) - .5) + .5);

    // specular
    float aspect = sqrt(1 - material.anisotropic * .9);
    float ax = max(.001, sqr(material.roughness) / aspect);
    float ay = max(.001, sqr(material.roughness) * aspect);
    float Ds = GTR2_aniso(NdotH, dot(H, X), dot(H, Y), ax, ay);
    float FH = SchlickFresnel(LdotH);
    float3 Fs = lerp(Cspec0, float3(1.0, 1.0, 1.0), FH);
    float Gs;
    Gs = smithG_GGX_aniso(NdotL, dot(L, X), dot(L, Y), ax, ay);
    Gs *= smithG_GGX_aniso(NdotV, dot(V, X), dot(V, Y), ax, ay);

    // sheen
    float3 Fsheen = FH * material.sheen * Csheen;

    // clearcoat (ior = 1.5 -> F0 = 0.04)
    float Dr = GTR1(NdotH, lerp(.1, .001, material.clearcoat_gloss));
    float Fr = lerp(.04, 1.0, FH);
    float Gr = smithG_GGX(NdotL, .25) * smithG_GGX(NdotV, .25);

    return ((1 / PI) * lerp(Fd, ss, material.subsurface) * Cdlin + Fsheen)
        * (1 - material.metallic)
        + Gs * Fs * Ds + .25 * material.clearcoat * Gr * Fr * Dr;
}


float3 BRDF(float3 L, float3 V, float3 N, Material material)
{
    switch (material.material_type)
    {
        case Material_Matte:
            return MatteBRDF(L, V, N, material);
        case Material_Plastic:
        case Material_Metal:
        case Material_Disney:
            return DisneyBRDF(L, V, N, material);
    }
    
    return float3(0.0, 0.0, 0.0);
}


[numthreads(LOCAL_SIZE, LOCAL_SIZE, 1)]
void main(CSParam param)
{
    uint2 extent;
    Lighting.GetDimensions(extent.x, extent.y);
    
    if (param.DispatchThreadID.x > extent.x || param.DispatchThreadID.y > extent.y)
    {
        return;
    }
    
    float2 uv = (float2(param.DispatchThreadID.xy) + float2(0.5, 0.5)) / float2(extent);
    
    float4 gbuffer0 = GBuffer0.Load(uint3(param.DispatchThreadID.xy, 0.0)); // GBuffer0: RGB - Albedo, A - metallic
    float4 gbuffer1 = GBuffer1.Load(uint3(param.DispatchThreadID.xy, 0.0)); // GBuffer1: RGA - normal, A - linear depth
    float4 gbuffer2 = GBuffer2.Load(uint3(param.DispatchThreadID.xy, 0.0)); // GBuffer2: RGB - emissive, A - roughness
    float4 gbuffer3 = GBuffer3.Load(uint3(param.DispatchThreadID.xy, 0.0)); // GBuffer3: R - entity id, G - instance id, BA - motion vector
    float depth = DepthBuffer.Load(uint3(param.DispatchThreadID.xy, 0.0)).r;
    
    MaterialData material_data = materials[gbuffer3.g];
    
    Material material;
    material.base_color = float4(gbuffer0.rgb, 1.0);
    material.metallic = gbuffer0.a;
    material.roughness = gbuffer2.a;
    material.emissive = gbuffer2.rgb;
    material.subsurface = material_data.subsurface;
    material.specular = material_data.specular;
    material.specular_tint = material_data.specular_tint;
    material.anisotropic = material_data.anisotropic;
    material.sheen = material_data.sheen;
    material.sheen_tint = material_data.sheen_tint;
    material.clearcoat = material_data.clearcoat;
    material.clearcoat_gloss = material_data.clearcoat_gloss;
    material.material_type = material_data.material_type;
    
    float3 frag_pos = WorldPositionFromDepth(uv, depth, mul(camera.inverse_view, camera.inverse_projection));
    float3 V = normalize(camera.position - frag_pos);
    float3 N = gbuffer1.rgb;
    
    float3 radiance = float3(0.0, 0.0, 0.0);
    
    // Handle Point Light
    for (uint i = 0; i < push_constants.point_light_count; i++)
    {
        PointLight light = point_lights[i];
        float3 L;
        float3 Li = light.Li(frag_pos, L);
        float3 f = BRDF(L, V, N, material);
        radiance += Li * f * abs(dot(L, N));
    }
    
    for (i = 0; i < push_constants.directional_light_count; i++)
    {
        DirectionalLight light = directional_lights[i];
        float3 L;
        float3 Li = light.Li(frag_pos, L);
        float3 f = BRDF(L, V, N, material);
        radiance += Li * f * abs(dot(L, N));
    }
    
    for (i = 0; i < push_constants.spot_light_count; i++)
    {
        SpotLight light = spot_lights[i];
        float3 L;
        float3 Li = light.Li(frag_pos, L);
        float3 f = BRDF(L, V, N, material);
        radiance += Li * f * abs(dot(L, N));
    }
   
    Lighting[int2(param.DispatchThreadID.xy)] = float4(radiance, 1.0);
}