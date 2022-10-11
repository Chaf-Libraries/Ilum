#ifdef MATERIAL_COMPILATION

#ifndef MATERIAL_HLSLI
#define MATERIAL_HLSLI

{{#Include}}
{{&IncludePath}}
{{/Include}}

{{#BxDFDefitions}}
{{&Definition}}
{{/BxDFDefitions}}

struct Material
{
    {{&BxDFType}} bxdfs;

    static Material Create()
    {
        Material material;
        material.bxdfs = {{&BxDFType}}::Create();
        return material;
    }

    float3 Eval(float3 wi, float3 wo)
    {
        return bxdfs.Eval(wi, wo);
    }
    
    float Pdf(float3 wi, float3 wo)
    {
        return bxdfs.Pdf(wi, wo);
    }
    
    float3 Samplef(float3 wi, float sample1, float2 sample2, out float3 wo, out float pdf)
    {
        return bxdfs.Samplef(wi, sample1, sample2. wo, pdf);
    }
};

#endif

#else

#ifndef MATERIAL_HLSLI
#define MATERIAL_HLSLI
struct Material
{
    static Material Create()
    {
        Material material;
        return material;
    }

    float3 Eval(float3 wi, float3 wo)
    {
        return 0.f;
    }
    
    float Pdf(float3 wi, float3 wo)
    {
        return 0.f;
    }
    
    float3 Samplef(float3 wi, float sample1, float2 sample2, out float3 wo, out float pdf)
    {
        return 0.f;
    }
};
#endif

#endif
