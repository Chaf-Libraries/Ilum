#include "../../Common.hlsli"
#include "../../Random.hlsli"
#include "../../Math.hlsli"
#include "../../SphericalHarmonic.hlsli"
#include "../../Light.hlsli"

#define LOCAL_SIZE 32

ConstantBuffer<Camera> camera : register(b0);
SamplerState ShadowMapSampler : register(s1);
SamplerState TextureSampler : register(s2);
Texture2D GBuffer0 : register(t3);
Texture2D GBuffer1 : register(t4);
Texture2D GBuffer2 : register(t5);
Texture2D GBuffer3 : register(t6);
Texture2D DepthBuffer : register(t7);
StructuredBuffer<MaterialData> materials : register(t8);
StructuredBuffer<DirectionalLight> directional_lights : register(t9);
StructuredBuffer<PointLight> point_lights : register(t10);
StructuredBuffer<SpotLight> spot_lights : register(t11);
Texture2DArray ShadowMaps : register(t12);
Texture2DArray CascadeShadowMaps : register(t13);
TextureCubeArray OmniShadowMaps : register(t14);
Texture2D EmuLut : register(t15);
Texture2D EavgLut : register(t16);
Texture2D IrradianceSH : register(t17);
TextureCube PrefilterMap : register(t18);
Texture2D BRDFPreIntegrate : register(t19);
RWTexture2D<float4> Lighting : register(u20);

[[vk::push_constant]]
struct
{
    uint directional_light_count;
    uint spot_light_count;
    uint point_light_count;
    uint enable_multi_bounce;
} push_constants;

static const int Sample_Method_Uniform = 0;
static const int Sample_Method_Poisson = 1;

static const int Shadow_Mode_None = 0;
static const int Shadow_Mode_Hard = 1;
static const int Shadow_Mode_PCF = 2;
static const int Shadow_Mode_PCSS = 3;

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

float LinearizeDepth(float depth, float znear, float zfar)
{
    float z = depth * 2.0 - 1.0;
    return znear * zfar / (zfar + depth * (znear - zfar));
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

float3 AverageFresnel(float3 r, float3 g)
{
    return float3(0.087237, 0.087237, 0.087237) + 0.0230685 * g - 0.0864902 * g * g + 0.0774594 * g * g * g + 0.782654 * r - 0.136432 * r * r + 0.278708 * r * r * r + 0.19744 * g * r + 0.0360605 * g * g * r - 0.2586 * g * r * r;
}

float3 MultiScatterBRDF(float3 albedo, float3 Eo, float3 Ei, float Eavg)
{
	// copper
    float3 edgetint = float3(0.827, 0.792, 0.678);
    float3 F_avg = AverageFresnel(albedo, edgetint);

    float3 f_add = F_avg * Eavg / (1.0 - F_avg * (1.0 - Eavg));
    float3 f_ms = (1.0 - Eo) * (1.0 - Ei) / (PI * (1.0 - Eavg.r));

    return f_ms * f_add;
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

    float3 Ctint = Cdlum > 0.0 ? Cdlin / Cdlum : float3(1.0, 1.0, 1.0); // normalize lum. to isolate hue+sat
    float3 Cspec0 = lerp(material.specular * 0.08 * lerp(float3(1.0, 1.0, 1.0), Ctint, material.specular_tint), Cdlin, material.metallic);
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
    float ss = 1.25 * (Fss * (1 / (NdotL + NdotV) - 0.5) + 0.5);

    // specular
    float aspect = sqrt(1 - material.anisotropic * .9);
    float ax = max(0.001, sqr(material.roughness) / aspect);
    float ay = max(0.001, sqr(material.roughness) * aspect);
    float Ds = GTR2_aniso(NdotH, dot(H, X), dot(H, Y), ax, ay);
    float FH = SchlickFresnel(LdotH);
    float3 Fs = lerp(Cspec0, float3(1.0, 1.0, 1.0), FH);
    float Gs;
    Gs = smithG_GGX_aniso(NdotL, dot(L, X), dot(L, Y), ax, ay);
    Gs *= smithG_GGX_aniso(NdotV, dot(V, X), dot(V, Y), ax, ay);

    // sheen
    float3 Fsheen = FH * material.sheen * Csheen;

    // clearcoat (ior = 1.5 -> F0 = 0.04)
    float Dr = GTR1(NdotH, lerp(0.1, 0.001, material.clearcoat_gloss));
    float Fr = lerp(0.04, 1.0, FH);
    float Gr = smithG_GGX(NdotL, 0.25) * smithG_GGX(NdotV, 0.25);
    
    float3 Fms = float3(0.0, 0.0, 0.0);
    if (push_constants.enable_multi_bounce)
    {
        float3 Eo = EmuLut.SampleLevel(TextureSampler, float2(dot(N, L), material.roughness), 0.0).rrr;
        float3 Ei = EmuLut.SampleLevel(TextureSampler, float2(dot(N, V), material.roughness), 0.0).rrr;
        float Eavg = EavgLut.SampleLevel(TextureSampler, float2(0.0, material.roughness), 0.0).r;
        
        Fms = MultiScatterBRDF(pow(material.base_color.rgb, float3(2.2, 2.2, 2.2)), Eo, Ei, Eavg);
    }

    return ((1.0 / PI) * lerp(Fd, ss, material.subsurface) * Cdlin + Fsheen) * (1.0 - material.metallic) + Gs * Fs * Ds + 0.25 * material.clearcoat * Gr * Fr * Dr + Fms;
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

// PCSS find block
float FindBlock(Texture2DArray shadowmap, float4 shadow_coord, float layer, float filter_scale, int filter_sample, int sample_method)
{
    uint2 tex_dim;
    uint layers;
    shadowmap.GetDimensions(tex_dim.x, tex_dim.y, layers);

    // Find blocker
    float z_blocker = 0.0;
    float num_blockers = 0.0;
    float2 offset = float2(0.0, 0.0);
    for (int i = 0; i < filter_sample; i++)
    {
        if (sample_method == Sample_Method_Uniform)
        {
            // Uniform sampling
            offset = UniformDiskSamples2D(shadow_coord.xy + offset);
        }
        else if (sample_method == Sample_Method_Poisson)
        {
            // Poisson sampling
            offset = PoissonDiskSamples2D(shadow_coord.xy + offset, filter_sample, 10, i);
        }
        
        offset = offset * filter_scale / float2(tex_dim);
        float dist = shadowmap.SampleLevel(ShadowMapSampler, float3(shadow_coord.xy + offset, layer), 0.0).r;
        if (dist < shadow_coord.z)
        {
            num_blockers += 1.0;
            z_blocker += dist;
        }
    }

    if (num_blockers == 0.0)
    {
        return 0.0;
    }

    return num_blockers == 0.0 ? 0.0 : z_blocker / num_blockers;
}

// PCSS find block cube
float FindBlockCube(TextureCubeArray shadowmap, float3 L, float layer, float filter_scale, int filter_sample, int sample_method)
{
    uint2 tex_dim;
    uint layers;
    shadowmap.GetDimensions(tex_dim.x, tex_dim.y, layers);
    
    float light_depth = length(L);

    // Find blocker
    float z_blocker = 0.0;
    float num_blockers = 0.0;
    float disk_radius = filter_scale / 100.0;
    float3 offset = float3(0.0, 0.0, 0.0);

    if (sample_method == 0)
    {
        for (int i = 0; i < filter_sample; i++)
        {
            // Uniform sampling
            offset = UniformDiskSamples3D(L + offset) * disk_radius;
            float dist = shadowmap.SampleLevel(ShadowMapSampler, float4(L + offset, layer), 0.0).r;
            if (light_depth > dist)
            {
                num_blockers += 1.0;
                z_blocker += dist;
            }
        }
    }
    else if (sample_method == 1)
    {
        int x = int(sqrt(filter_sample));
        int y = filter_sample / x;
        for (int i = 0; i < x; i++)
        {
            for (int j = 0; j < y; j++)
            {
                // Poisson sampling
                offset = PoissonDiskSamples3D(L + offset, x * y, 10, float2(i, j)) * disk_radius;
                float dist = shadowmap.SampleLevel(ShadowMapSampler, float4(L + offset, layer), 0.0).r;
                if (light_depth > dist)
                {
                    num_blockers += 1.0;
                    z_blocker += dist;
                }
            }
        }
    }

    if (num_blockers == 0.0)
    {
        return 0.0;
    }

    return num_blockers == 0.0 ? 0.0 : z_blocker / num_blockers;
}

// Sample shadow map
float SampleShadowmap(Texture2DArray shadowmap, float4 shadow_coord, float layer, float2 offset)
{
    float shadow = 1.0;
	
    if (shadow_coord.z > -1.0 && shadow_coord.z < 1.0)
    {
        float dist = shadowmap.SampleLevel(ShadowMapSampler, float3(shadow_coord.xy + offset, layer), 0.0).r;
        if (shadow_coord.w > 0.0 && dist < shadow_coord.z)
        {
            shadow = 0.0;
        }
    }
    return shadow;
}

// Sample shadow map via PCF
float SampleShadowmapPCF(Texture2DArray shadowmap, float4 shadow_coord, float layer, int filter_sample, float filter_scale, int sample_method)
{
    uint2 tex_dim;
    uint layers;
    shadowmap.GetDimensions(tex_dim.x, tex_dim.y, layers);

    float shadow_factor = 0.0;

    float2 offset = float2(0.0, 0.0);
    for (int i = 0; i < filter_sample; i++)
    {
        if (sample_method == Sample_Method_Uniform)
        {
            // Uniform sampling
            offset = UniformDiskSamples2D(shadow_coord.xy + offset);
        }
        else if (sample_method == Sample_Method_Poisson)
        {
            // Poisson sampling
            offset = PoissonDiskSamples2D(shadow_coord.xy + offset, filter_sample, 10, i);
        }
        offset = offset * filter_scale / float2(tex_dim);
        shadow_factor += SampleShadowmap(shadowmap, shadow_coord, layer, offset);
    }

    return shadow_factor / float(filter_sample);
}

// Sample shadow map via PCSS
float SampleShadowmapPCSS(Texture2DArray shadowmap, float4 shadow_coord, float layer, float filter_scale, int filter_sample, int sample_method, float light_size)
{
    uint2 tex_dim;
    uint layers;
    shadowmap.GetDimensions(tex_dim.x, tex_dim.y, layers);
    
    float z_receiver = LinearizeDepth(shadow_coord.z, 0.01, 1000.0);

    // Penumbra size
    float z_blocker = LinearizeDepth(FindBlock(shadowmap, shadow_coord, layer, filter_scale, filter_sample, sample_method), 0.01, 1000.0);
    float w_light = 0.1;
    float w_penumbra = (z_receiver - z_blocker) * light_size / z_blocker;

    // Filtering
    float shadow_factor = 0.0;
    float2 offset = float2(0.0, 0.0);
    for (int i = 0; i < filter_sample; i++)
    {
        if (sample_method == Sample_Method_Uniform)
        {
            // Uniform sampling
            offset = UniformDiskSamples2D(shadow_coord.xy + offset);
        }
        else if (sample_method == Sample_Method_Poisson)
        {
            // Poisson sampling
            offset = PoissonDiskSamples2D(shadow_coord.xy + offset, filter_sample, 10, i);
        }
        offset = offset * w_penumbra / float2(tex_dim);
        shadow_factor += SampleShadowmap(shadowmap, shadow_coord, layer, offset);
    }

    return shadow_factor / float(filter_sample);
}

// Sample shadow cubemap
float SampleShadowmapCube(TextureCubeArray shadowmap, float3 L, float layer, float3 offset)
{
    float shadow = 1.0;
    float light_depth = length(L);
    L.z = -L.z;
    // Reconstruct depth
    float dist = shadowmap.SampleLevel(ShadowMapSampler, float4(L + offset, layer), 0.0).r;
    dist *= 100.0;

    if (light_depth > dist)
    {
        shadow = 0.0;
    }

    return shadow;
}

// Sample shadow cubemap via PCF
float SampleShadowmapCubePCF(TextureCubeArray shadowmap, float3 L, float layer, float filter_scale, int filter_sample, int sample_method)
{
    uint2 tex_dim;
    uint layers;
    shadowmap.GetDimensions(tex_dim.x, tex_dim.y, layers);

    float shadow_factor = 0.0;
    float light_depth = length(L);
    
    float disk_radius = filter_scale / 100.0;

    float3 offset = float3(0.0, 0.0, 0.0);
    int count = 0;
    if (sample_method == 0)
    {
        for (int i = 0; i < filter_sample; i++)
        {
            // Uniform sampling
            offset = UniformDiskSamples3D(L + offset) * disk_radius;
            shadow_factor += SampleShadowmapCube(shadowmap, L, layer, offset);
        }
        count = filter_sample;
    }
    else if (sample_method == 1)
    {
        int x = int(sqrt(filter_sample));
        int y = filter_sample / x;
        count = x * y;
        for (int i = 0; i < x; i++)
        {
            for (int j = 0; j < y; j++)
            {
                // Poisson sampling
                offset = PoissonDiskSamples3D(L + offset, count, 10, float2(i, j)) * disk_radius;
                shadow_factor += SampleShadowmapCube(shadowmap, L, layer, offset);
            }
        }
    }

    return shadow_factor / float(count);
}

// Sample shadow cubemap via PCSS
float SampleShadowmapCubePCSS(TextureCubeArray shadowmap, float3 L, float layer, float filter_scale, int filter_sample, int sample_method, float light_size)
{
    uint2 tex_dim;
    uint layers;
    shadowmap.GetDimensions(tex_dim.x, tex_dim.y, layers);
    
    float light_depth = length(L);
    float z_receiver = LinearizeDepth(light_depth, 0.01, 1000.0);

    // Penumbra size
    float z_blocker = LinearizeDepth(FindBlockCube(shadowmap, L, layer, filter_scale, filter_sample, sample_method), 0.01, 1000.0);
    float w_light = 0.1;
    float w_penumbra = (z_receiver - z_blocker) * light_size / z_blocker;

    // Filtering
    float shadow_factor = 0.0;
    float3 offset = float3(0.0, 0.0, 0.0);
    float disk_radius = filter_scale / 100.0;
    int count = 0;

    if (sample_method == 0)
    {
        for (int i = 0; i < filter_sample; i++)
        {
            // Uniform sampling
            offset = UniformDiskSamples3D(L + offset) * w_penumbra / float(tex_dim.x);
            shadow_factor += SampleShadowmapCube(shadowmap, L, layer, offset);
        }
        count = filter_sample;
    }
    else if (sample_method == 1)
    {
        int x = int(sqrt(filter_sample));
        int y = filter_sample / x;
        count = x * y;
        for (int i = 0; i < x; i++)
        {
            for (int j = 0; j < y; j++)
            {
                // Poisson sampling
                offset = PoissonDiskSamples3D(L + offset, count, 10, float2(i, j)) * w_penumbra / float(tex_dim.x);
                shadow_factor += SampleShadowmapCube(shadowmap, L, layer, offset);
            }
        }
    }
    return shadow_factor / float(filter_sample);
}

// Shadow for spot light
float3 SpotLightShadow(SpotLight light, float3 frag_color, float3 frag_pos, float layer)
{
    float4 shadow_clip = mul(light.view_projection, float4(frag_pos, 1.0));
    float4 shadow_coord = float4(shadow_clip.xyz / shadow_clip.w, shadow_clip.w);
    shadow_coord.xy = shadow_coord.xy * 0.5 + 0.5;
    shadow_coord.y = 1.0 - shadow_coord.y;

    switch (light.shadow_mode)
    {
        case Shadow_Mode_None:
            return frag_color;
        case Shadow_Mode_Hard:
            return frag_color * SampleShadowmap(ShadowMaps, shadow_coord, layer, float2(0.0, 0.0));
        case Shadow_Mode_PCF:
            return frag_color * SampleShadowmapPCF(ShadowMaps, shadow_coord, layer, light.filter_sample, light.filter_scale, light.sample_method);
        case Shadow_Mode_PCSS:
            return frag_color * SampleShadowmapPCSS(ShadowMaps, shadow_coord, layer, light.filter_scale, light.filter_sample, light.sample_method, light.radius);
    }
    
    return frag_color;
}

// Shadow for directional light
float3 DirectionalLightShadow(DirectionalLight light, float3 frag_color, float3 frag_pos, float linear_z, float layer)
{
    uint cascade_index = 0;
    // Select cascade
    for (uint i = 0; i < 3; ++i)
    {
        if (light.split_depth[i] > -linear_z)
        {
            cascade_index = i + 1;
        }
    }

    float4 shadow_clip = mul(light.view_projection[cascade_index], float4(frag_pos, 1.0));
    float4 shadow_coord = float4(shadow_clip.xyz / shadow_clip.w, shadow_clip.w);
    shadow_coord.xy = shadow_coord.xy * 0.5 + 0.5;
    shadow_coord.y = 1.0 - shadow_coord.y;

    layer = layer * 4 + cascade_index;
    switch (light.shadow_mode)
    {
        case Shadow_Mode_None:
            return frag_color;
        case Shadow_Mode_Hard:
            return frag_color * SampleShadowmap(CascadeShadowMaps, shadow_coord, layer, float2(0.0, 0.0));
        case Shadow_Mode_PCF:
            return frag_color * SampleShadowmapPCF(CascadeShadowMaps, shadow_coord, layer, light.filter_sample, light.filter_scale, light.sample_method);
        case Shadow_Mode_PCSS:
            return frag_color * SampleShadowmapPCSS(CascadeShadowMaps, shadow_coord, layer, light.filter_scale, light.filter_sample, light.sample_method, light.radius);
    }

    return frag_color;
}

// Shadow for point light
float3 PointLightShadow(PointLight light, float3 frag_color, float3 frag_pos, float layer)
{
    float3 L = frag_pos - light.position;
   
    switch (light.shadow_mode)
    {
        case Shadow_Mode_None:
            return frag_color;
        case Shadow_Mode_Hard:
            return frag_color * SampleShadowmapCube(OmniShadowMaps, L, layer, float3(0.0, 0.0, 0.0));
        case Shadow_Mode_PCF:
            return frag_color * SampleShadowmapCubePCF(OmniShadowMaps, L, layer, light.filter_scale, light.filter_sample, light.sample_method);
        case Shadow_Mode_PCSS:
            return frag_color * SampleShadowmapCubePCSS(OmniShadowMaps, L, layer, light.filter_scale, light.filter_sample, light.sample_method, light.radius);
    }

    return frag_color;
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
    float4 gbuffer1 = GBuffer1.Load(uint3(param.DispatchThreadID.xy, 0.0)); // GBuffer1: RGB - normal, A - linear depth
    float4 gbuffer2 = GBuffer2.Load(uint3(param.DispatchThreadID.xy, 0.0)); // GBuffer2: RGB - emissive, A - roughness
    float4 gbuffer3 = GBuffer3.Load(uint3(param.DispatchThreadID.xy, 0.0)); // GBuffer3: R - entity id, G - instance id, BA - motion vector
    float depth = DepthBuffer.Load(uint3(param.DispatchThreadID.xy, 0.0)).r;
    
    float linear_z = gbuffer1.a;
    
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
    
    float3 radiance = material.emissive;
    
    // Handle point light
    for (uint i = 0; i < push_constants.point_light_count; i++)
    {
        PointLight light = point_lights[i];
        float3 L;
        float3 Li = light.Li(frag_pos, L);
        float3 f = BRDF(L, V, N, material);
        radiance += PointLightShadow(light, Li * f * abs(dot(L, N)), frag_pos, i);
    }
    
    // Handle directional light
    for (i = 0; i < push_constants.directional_light_count; i++)
    {
        DirectionalLight light = directional_lights[i];
        float3 L;
        float3 Li = light.Li(frag_pos, L);
        float3 f = BRDF(L, V, N, material);
        radiance += DirectionalLightShadow(light, Li * f * abs(dot(L, N)), frag_pos, linear_z, i);
    }
    
    // Handle spot light
    for (i = 0; i < push_constants.spot_light_count; i++)
    {
        SpotLight light = spot_lights[i];
        float3 L;
        float3 Li = light.Li(frag_pos, L);
        float3 f = BRDF(L, V, N, material);
        radiance += SpotLightShadow(light, Li * f * abs(dot(L, N)), frag_pos, i);
    }
    
    // Handle environment light
    {
        float3 F0 = float3(0.0, 0.0, 0.0);
        F0 = lerp(F0, material.base_color.rgb, material.metallic);
        float3 F = F0 + (max(float3(1.0 - material.roughness, 1.0 - material.roughness, 1.0 - material.roughness), F0) * pow(1.0 - max(dot(N, V), 0.0), 5.0));
        float3 Kd = (1.0 - F) * (1.0 - material.metallic);

        float3 irradiance = float3(0.0, 0.0, 0.0);
        SH9 basis = EvaluateSH(N);
        for (uint i = 0; i < 9; i++)
        {
            irradiance += IrradianceSH.Load(uint3(i, 0, 0)).rgb * basis.weights[i];
        }
        irradiance = max(float3(0.0, 0.0, 0.0), irradiance) * InvPI;
        
        float3 diffuse = irradiance * material.base_color.rgb;

        const float MAX_PREFILTER_MAP_LOD = 4.0;
        float3 prefiltered_color = PrefilterMap.SampleLevel(TextureSampler, reflect(-V, N), material.roughness * MAX_PREFILTER_MAP_LOD).rgb;
        float2 brdf = BRDFPreIntegrate.SampleLevel(TextureSampler, float2(clamp(dot(N, V), 0.0, 1.0), material.roughness), 0.0).rg;
        float3 specular = prefiltered_color * (F * brdf.x + brdf.y);

        float3 ambient = Kd * diffuse + specular;
        radiance += ambient;
    }

   
    Lighting[int2(param.DispatchThreadID.xy)] = float4(radiance, 1.0);
}