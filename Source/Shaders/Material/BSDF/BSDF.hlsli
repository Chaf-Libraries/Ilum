#ifndef BSDF_HLSLI
#define BSDF_HLSLI

// BxDF Flags
static uint BSDF_Reflection = 1 << 0;
static uint BSDF_Transmission = 1 << 1;
static uint BSDF_Diffuse = 1 << 2;
static uint BSDF_Glossy = 1 << 3;
static uint BSDF_Specular = 1 << 4;
static uint BSDF_DiffuseReflection = BSDF_Diffuse | BSDF_Reflection;
static uint BSDF_DiffuseTransmission = BSDF_Diffuse | BSDF_Transmission;
static uint BSDF_GlossyReflection = BSDF_Glossy | BSDF_Reflection;
static uint BSDF_GlossyTransmission = BSDF_Glossy | BSDF_Transmission;
static uint BSDF_SpecularReflection = BSDF_Specular | BSDF_Reflection;
static uint BSDF_SpecularTransmission = BSDF_Specular | BSDF_Transmission;
static uint BSDF_All = BSDF_Diffuse | BSDF_Glossy | BSDF_Specular | BSDF_Reflection | BSDF_Transmission;

// Sample Flag
static uint SampleFlags_Unset = 0;
static uint SampleFlags_Reflection = 1 << 0;
static uint SampleFlags_Transmission = 1 << 1;
static uint SampleFlags_All = SampleFlags_Reflection | SampleFlags_Transmission;

// Transport Mode
static uint TransportMode_Radiance = 0;
static uint TransportMode_Importance = 1;

#define SampleFlags uint
#define TransportMode uint
#define BxDFFlags uint

bool IsReflective(uint f)
{
    return f & BSDF_Reflection;
}

bool IsTransmissive(uint f)
{
    return f & BSDF_Transmission;
}

bool IsDiffuse(uint f)
{
    return f & BSDF_Diffuse;
}

bool IsGlossy(uint f)
{
    return f & BSDF_Glossy;
}

bool IsSpecular(uint f)
{
    return f & BSDF_Specular;
}

bool IsNonSpecular(uint f)
{
    return f & (BSDF_Diffuse | BSDF_Glossy);
}

struct BSDFSample
{
    float3 f;
    float3 wiW;
    float3 wi;
    float pdf;
    BxDFFlags flags;
    float eta;
    bool pdfIsProportional;
    
    void Init()
    {
        f = 0.f;
        wiW = 0.f;
        wi = 0.f;
        pdf = 0.f;
        flags = 0.f;
        eta = 1.f;
        pdfIsProportional = false;

    }
    
    bool IsReflection()
    {
        return flags & BSDF_Reflection;
    }
    
    bool IsTransmission()
    {
        return flags & BSDF_Transmission;
    }
    
    bool IsDiffuse()
    {
        return flags & BSDF_Diffuse;
    }
    
    bool IsGlossy()
    {
        return flags & BSDF_Glossy;
    }
    
    bool IsSpecular()
    {
        return flags & BSDF_Specular;
    }
};

float3 SRGBtoLINEAR(float3 srgb_in)
{
#ifdef SRGB_FAST_APPROXIMATION
  float3 linOut = pow(srgbIn, vec3(2.2));
#else  //SRGB_FAST_APPROXIMATION
  float3 bLess  = step(0.04045, srgb_in);
  float3 linOut = lerp(srgb_in.xyz / 12.92, pow((srgb_in + 0.055) / 1.055, 2.4), bLess);
#endif  //SRGB_FAST_APPROXIMATION
  return linOut;
}

// struct BSDF
// {
//     void Init();

//     float3 Eval(float3 wo, float3 wi, TransportMode mode);

//     float PDF(float3 wo, float3 wi, TransportMode mode, SampleFlags flags);

//     BSDFSample Samplef(float3 wo, float uc, float2 u, TransportMode mode, SampleFlags flags);

//     float3 GetEmissive();

// };

#define HAS_NO_EMISSIVE float3 GetEmissive(){ return 0.f; }

#endif