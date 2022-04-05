#ifndef __MATERIAL_HLSL__
#define __MATERIAL_HLSL__

#define TEXTURE_BASE_COLOR 0
#define TEXTURE_NORMAL 1
#define TEXTURE_METALLIC 2
#define TEXTURE_ROUGHNESS 3
#define TEXTURE_EMISSIVE 4
#define TEXTURE_AO 5
#define TEXTURE_DISPLACEMENT 6
#define TEXTURE_MAX_NUM 7

#define MAX_TEXTURE_ARRAY_SIZE 1024

#define BxDF_CookTorrance 0
#define BxDF_Disney 1
#define BxDF_Matte 2
#define BxDF_Plastic 3
#define BxDF_Metal 4
#define BxDF_Mirror 5
#define BxDF_Substrate 6
#define BxDF_Glass 7

struct MaterialData
{
    float4 base_color;

    float3 emissive_color;
    float emissive_intensity;

    float displacement;
    float subsurface;
    float metallic;
    float specular;

    float specular_tint;
    float roughness;
    float anisotropic;
    float sheen;

    float sheen_tint;
    float clearcoat;
    float clearcoat_gloss;
    float transmission;

    float transmission_roughness;
    uint textures[TEXTURE_MAX_NUM];

    float3 data;
    uint material_type;
};

struct Material
{
    float4 base_color;
    float3 emissive;
    float subsurface;
    float metallic;
    float specular;
    float specular_tint;
    float roughness;
    float anisotropic;
    float sheen;
    float sheen_tint;
    float clearcoat;
    float clearcoat_gloss;
    float transmission;
    float3 data;
    uint material_type;
};
#endif