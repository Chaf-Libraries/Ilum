#include "../../Common.hlsli"
#include "../../Random.hlsli"
#include "../../Math.hlsli"
#include "../../SphericalHarmonic.hlsli"
#include "../../Light.hlsli"

RWTexture2D<uint> VBuffer : register(u0);

Texture2D textureArray[] : register(t1);
SamplerState TextureSampler : register(s1);
SamplerState ShadowSampler : register(s2);

ConstantBuffer<Camera> camera : register(b3);
StructuredBuffer<MaterialData> materials : register(t4);
StructuredBuffer<Instance> instances : register(t5);
StructuredBuffer<Meshlet> meshlets : register(t6);

StructuredBuffer<Vertex> vertices : register(t7);
StructuredBuffer<uint> meshlet_vertices : register(t8);
StructuredBuffer<uint> meshlet_indices : register(t9);

StructuredBuffer<DirectionalLight> directional_lights : register(t10);
StructuredBuffer<PointLight> point_lights : register(t11);
StructuredBuffer<SpotLight> spot_lights : register(t12);

Texture2DArray ShadowMaps : register(t13);
Texture2DArray CascadeShadowMaps : register(t14);
TextureCubeArray OmniShadowMaps : register(t15);

Texture2D EmuLut : register(t16);
Texture2D EavgLut : register(t17);
Texture2D IrradianceSH : register(t18);
TextureCube PrefilterMap : register(t19);
Texture2D BRDFPreIntegrate : register(t20);

RWTexture2D<float4> Lighting : register(u21);
RWTexture2D<float2> Normal : register(u22);

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

void ComputeBarycentrics(float2 uv, float4 v0, float4 v1, float4 v2, out float3 bary, out float3 bary_ddx, out float3 bary_ddy)
{
    float3 pos0 = v0.xyz / v0.w;
    float3 pos1 = v1.xyz / v1.w;
    float3 pos2 = v2.xyz / v2.w;
    
    float3 rcp_w = rcp(float3(v0.w, v1.w, v2.w));
    
    float3 pos120X = float3(pos1.x, pos2.x, pos0.x);
    float3 pos120Y = float3(pos1.y, pos2.y, pos0.y);
    float3 pos201X = float3(pos2.x, pos0.x, pos1.x);
    float3 pos201Y = float3(pos2.y, pos0.y, pos1.y);
    
    float3 C_dx = pos201Y - pos120Y;
    float3 C_dy = pos120X - pos201X;
    
    float3 C = C_dx * (uv.x - pos120X) + C_dy * (uv.y - pos120Y);
    float3 G = C * rcp_w;
    
    float H = dot(C, rcp_w);
    float rcpH = rcp(H);
    
    bary = G * rcpH;
    
    float3 G_dx = C_dx * rcp_w;
    float3 G_dy = C_dy * rcp_w;

    float H_dx = dot(C_dx, rcp_w);
    float H_dy = dot(C_dy, rcp_w);
    
    uint2 extent;
    VBuffer.GetDimensions(extent.x, extent.y);
    
    bary_ddx = (G_dx * H - G * H_dx) * (rcpH * rcpH) * (2.0 / float(extent.x));
    bary_ddy = (G_dy * H - G * H_dy) * (rcpH * rcpH) * (-2.0 / float(extent.y));
}

struct VBufferAttribute
{
    float3 position;
    float2 uv;
    float3 normal;
    float3 tangent;
    float depth;
    
    float2 dx;
    float2 dy;
    float3 bary;
    
    uint matID;
};

uint hash(uint a)
{
    a = (a + 0x7ed55d16) + (a << 12);
    a = (a ^ 0xc761c23c) ^ (a >> 19);
    a = (a + 0x165667b1) + (a << 5);
    a = (a + 0xd3a2646c) ^ (a << 9);
    a = (a + 0xfd7046c5) + (a << 3);
    a = (a ^ 0xb55a4f09) ^ (a >> 16);
    return a;
}

VBufferAttribute GetVBufferAttribute(uint vbuffer_data, float2 uv)
{
    VBufferAttribute attribute;
    
    uint primitive_id;
    uint meshlet_id;
    
    UnPackVBuffer(vbuffer_data, meshlet_id, primitive_id);
    
    Meshlet meshlet = meshlets[meshlet_id];
    Instance instance = instances[meshlet.instance_id];
    
    attribute.matID = meshlet.instance_id;
    
    uint vertex_idx[3];
    
    for (int j = primitive_id * 3; j < primitive_id * 3 + 3; j++)
    {
        uint a = (meshlet.meshlet_index_offset + j) / 4;
        uint b = (meshlet.meshlet_index_offset + j) % 4;
        uint idx = (meshlet_indices[a] & 0x000000ffU << (8 * b)) >> (8 * b);
        vertex_idx[j % 3] = idx;
    }
    
    vertex_idx[0] = meshlet.vertex_offset + meshlet_vertices[meshlet.meshlet_vertex_offset + vertex_idx[0]];
    vertex_idx[1] = meshlet.vertex_offset + meshlet_vertices[meshlet.meshlet_vertex_offset + vertex_idx[1]];
    vertex_idx[2] = meshlet.vertex_offset + meshlet_vertices[meshlet.meshlet_vertex_offset + vertex_idx[2]];
    
    Vertex v0 = vertices[vertex_idx[0]];
    Vertex v1 = vertices[vertex_idx[1]];
    Vertex v2 = vertices[vertex_idx[2]];
    
    float4 clip_pos0 = mul(camera.view_projection, mul(instance.transform, float4(v0.position.xyz, 1.0)));
    float4 clip_pos1 = mul(camera.view_projection, mul(instance.transform, float4(v1.position.xyz, 1.0)));
    float4 clip_pos2 = mul(camera.view_projection, mul(instance.transform, float4(v2.position.xyz, 1.0)));

    float2 pixel_clip = uv * 2.0 - 1.0;
    pixel_clip.y *= -1;
    
    float3 barycentrics;
    float3 ddx_barycentrics;
    float3 ddy_barycentrics;
    
    ComputeBarycentrics(pixel_clip, clip_pos0, clip_pos1, clip_pos2, barycentrics, ddx_barycentrics, ddy_barycentrics);
    attribute.bary = barycentrics;
    attribute.depth = clip_pos0.z * barycentrics.x + clip_pos1.z * barycentrics.y + clip_pos2.z * barycentrics.z;
    attribute.uv = v0.uv.xy * barycentrics.x + v1.uv.xy * barycentrics.y + v2.uv.xy * barycentrics.z;
    attribute.position = v0.position.xyz * barycentrics.x + v1.position.xyz * barycentrics.y + v2.position.xyz * barycentrics.z;
    attribute.position = mul(instance.transform, float4(attribute.position, 1.0)).xyz;
    attribute.normal = v0.normal.xyz * barycentrics.x + v1.normal.xyz * barycentrics.y + v2.normal.xyz * barycentrics.z;
    attribute.normal = v0.normal.xyz * barycentrics.x + v1.normal.xyz * barycentrics.y + v2.normal.xyz * barycentrics.z;
    attribute.normal = normalize(mul((float3x3) instance.transform, attribute.normal));
    attribute.tangent = v0.tangent.xyz * barycentrics.x + v1.tangent.xyz * barycentrics.y + v2.tangent.xyz * barycentrics.z;
    attribute.tangent = normalize(mul((float3x3) instance.transform, attribute.tangent.xyz));
    attribute.dx = v0.uv.xy * ddx_barycentrics.x + v1.uv.xy * ddx_barycentrics.y + v2.uv.xy * ddx_barycentrics.z;
    attribute.dy = v0.uv.xy * ddy_barycentrics.x + v1.uv.xy * ddy_barycentrics.y + v2.uv.xy * ddy_barycentrics.z;
    
    return attribute;
}

Material GetMaterial(inout VBufferAttribute attribute)
{
    Material mat;
    
    MaterialData material = materials[attribute.matID];
    
    mat.base_color = material.base_color;
    mat.emissive = material.emissive_color * material.emissive_intensity;
    mat.subsurface = material.subsurface;
    mat.metallic = material.metallic;
    mat.specular = material.specular;
    mat.specular_tint = material.specular_tint;
    mat.roughness = material.roughness;
    mat.anisotropic = material.anisotropic;
    mat.sheen = material.sheen;
    mat.sheen_tint = material.sheen_tint;
    mat.clearcoat = material.clearcoat;
    mat.clearcoat_gloss = material.clearcoat_gloss;
    mat.specular_transmission = material.specular_transmission;
    mat.diffuse_transmission = material.diffuse_transmission;
    mat.refraction = material.refraction;
    mat.flatness = material.flatness;
    mat.thin = material.thin;
    mat.material_type = material.material_type;
    mat.data = material.data;

    if (material.textures[TEXTURE_BASE_COLOR] < MAX_TEXTURE_ARRAY_SIZE)
    {
        float4 base_color = textureArray[NonUniformResourceIndex(material.textures[TEXTURE_BASE_COLOR])].SampleGrad(TextureSampler, attribute.uv, attribute.dx, attribute.dy).rgba;
        base_color.xyz = pow(base_color.xyz, float3(2.2, 2.2, 2.2));
        mat.base_color.rgba *= base_color;
    }
    
    if (material.textures[TEXTURE_EMISSIVE] < MAX_TEXTURE_ARRAY_SIZE)
    {
        float3 emissive = textureArray[NonUniformResourceIndex(material.textures[TEXTURE_EMISSIVE])].SampleGrad(TextureSampler, attribute.uv, attribute.dx, attribute.dy).rgb;
        emissive = pow(emissive, float3(2.2, 2.2, 2.2));
        mat.emissive *= emissive;
    }
   
    if (material.textures[TEXTURE_METALLIC] < MAX_TEXTURE_ARRAY_SIZE)
    {
        float metallic = textureArray[NonUniformResourceIndex(material.textures[TEXTURE_METALLIC])].SampleGrad(TextureSampler, attribute.uv, attribute.dx, attribute.dy).r;
        mat.metallic *= metallic;
    }
     
    if (material.textures[TEXTURE_ROUGHNESS] < MAX_TEXTURE_ARRAY_SIZE)
    {
        float roughness = textureArray[NonUniformResourceIndex(material.textures[TEXTURE_ROUGHNESS])].SampleGrad(TextureSampler, attribute.uv, attribute.dx, attribute.dy).g;
        mat.roughness *= roughness;
    }
    
    if (material.textures[TEXTURE_NORMAL] < MAX_TEXTURE_ARRAY_SIZE)
    {
        float3 T = attribute.tangent;
        float3 B;
        if (!any(T))
        {
            CreateCoordinateSystem(attribute.normal, T, B);
        }
        else
        {
            float3 B = normalize(cross(attribute.normal, T));
        }
        
        float3x3 TBN = float3x3(T, B, attribute.normal);
        float3 normalVector = textureArray[NonUniformResourceIndex(material.textures[TEXTURE_NORMAL])].SampleGrad(TextureSampler, attribute.uv, attribute.dx, attribute.dy).rgb;
        normalVector = normalVector * 2.0 - 1.0;
        normalVector = normalize(normalVector);
        attribute.normal = normalize(mul(normalVector, TBN));
    }
    
    return mat;
}

#define sqr(x) x * x

float LinearizeDepth(float depth, float znear, float zfar)
{
    float z = depth * 2.0 - 1.0;
    return znear * zfar / (zfar + depth * (znear - zfar));
}

float DistributeGGX(float NoH, float roughness)
{
    float alpha = roughness * roughness;
    float alpha2 = alpha * alpha;
    float denom = NoH * NoH * (alpha2 - 1.0) + 1.0;
    return alpha2 / (PI * denom * denom);
}

float GeometrySchlickGGX(float NoV, float roughness)
{
    float r = (roughness + 1.0);
    float k = (r * r) / 8.0;
    return NoV / (NoV * (1.0 - k) + k);
}

float GeometrySmith(float NoL, float NoV, float roughness)
{
    float ggx1 = GeometrySchlickGGX(NoL, roughness);
    float ggx2 = GeometrySchlickGGX(NoV, roughness);

    return ggx1 * ggx2;
}

float3 FresnelSchlick(float LoH, float3 F0)
{
    return F0 + (1.0 - F0) * pow(clamp(1 - LoH, 0, 1), 5.0);
}

float3 LambertianDiffuse(float3 albedo)
{
    return albedo / PI;
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

/*float3 BRDF(float3 L, float3 V, float3 N, Material material)
{
    float3 Cdlin = pow(material.base_color.rgb, float3(2.2, 2.2, 2.2));
    float3 F0 = float3(0.04, 0.04, 0.04);
    F0 = lerp(F0, Cdlin, material.metallic);

    float3 H = normalize(V + L);
    float NoV = saturate(dot(N, V));
    float NoL = saturate(dot(N, L));
    float NoH = saturate(dot(N, H));
    float HoV = saturate(dot(H, V));

    float D = DistributeGGX(NoH, material.roughness);
    float G = GeometrySmith(NoL, NoV, material.roughness);
    float3 F = FresnelSchlick(HoV, F0);

    float3 specular = D * F * G / (4.0 * NoL * NoV + 0.001);
    float3 Kd = lerp(float3(1.0, 1.0, 1.0) - F, float3(0.0, 0.0, 0.0), material.metallic);
    
    float3 Fms = float3(0.0, 0.0, 0.0);
    
    if (push_constants.enable_multi_bounce)
    {
        float3 Eo = EmuLut.SampleLevel(TextureSampler, float2(dot(N, L), material.roughness), 0.0).rrr;
        float3 Ei = EmuLut.SampleLevel(TextureSampler, float2(dot(N, V), material.roughness), 0.0).rrr;
        float Eavg = EavgLut.SampleLevel(TextureSampler, float2(0.0, material.roughness), 0.0).r;
        
        Fms = MultiScatterBRDF(pow(material.base_color.rgb, float3(2.2, 2.2, 2.2)), Eo, Ei, Eavg);
    }

    return Kd * LambertianDiffuse(Cdlin) + specular + Fms;
}*/

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
        float dist = shadowmap.SampleLevel(ShadowSampler, float3(shadow_coord.xy + offset, layer), 0.0).r;
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
            float dist = shadowmap.SampleLevel(ShadowSampler, float4(L + offset, layer), 0.0).r;
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
                float dist = shadowmap.SampleLevel(ShadowSampler, float4(L + offset, layer), 0.0).r;
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
        float dist = shadowmap.SampleLevel(ShadowSampler, float3(shadow_coord.xy + offset, layer), 0.0).r;
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
    float dist = shadowmap.SampleLevel(ShadowSampler, float4(L + offset, layer), 0.0).r;
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

[numthreads(32, 32, 1)]
void main(CSParam param)
{
    uint2 extent;
    VBuffer.GetDimensions(extent.x, extent.y);

    if (param.DispatchThreadID.x > extent.x || param.DispatchThreadID.y > extent.y)
    {
        return;
    }

    float2 pixel_size = 1.0 / float2(extent);

    float2 uv = (float2(param.DispatchThreadID.xy + float2(0.5, 0.5))) / float2(extent);
    
    uint vdata = VBuffer[param.DispatchThreadID.xy];
    
    if (vdata == 0xffffffff)
    {
        Lighting[param.DispatchThreadID.xy] = float4(0.0, 0.0, 0.0, 1.0);
        Normal[param.DispatchThreadID.xy] = PackNormal(float3(0.0, 0.0, 0.0));
        return;
    }
    
    VBufferAttribute attribute = GetVBufferAttribute(vdata, uv);
    Material material = GetMaterial(attribute);
    
    float3 frag_pos = attribute.position;
        
    float3 V = normalize(camera.position - frag_pos);
    float3 N = attribute.normal;

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
        radiance += DirectionalLightShadow(light, Li * f * abs(dot(L, N)), frag_pos, attribute.depth, i);
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
        radiance += 0.000000000001 * ambient;
    }

    Lighting[param.DispatchThreadID.xy] = float4(radiance, material.base_color.a);
    Normal[param.DispatchThreadID.xy] = PackNormal(attribute.normal.rgb);
}