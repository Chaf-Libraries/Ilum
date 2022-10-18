#ifndef MATERIAL_HLSLI
#define MATERIAL_HLSLI

#ifdef MATERIAL_COMPILATION
{{#Headers}}
{{&Header}}
{{/Headers}}

struct Material
{
    {{#Declarations}}
    {{Declaration}}
    {{/Declarations}}

    {{#Functions}}
    {{Function}}
    {{/Functions}}

    void Init()
    {
        {{#Definitions}}
        {{Definition}}
        {{/Definitions}}
    }

    float3 _Eval(float3 wi, float3 wo)
    {
        return 0.f;
    }

    float _Pdf(float3 wi, float3 wo)
    {
        return 0.f;
    }
    
    float3 _Samplef(float3 wo, float uc, float2 u, out float3 wi, out float pdf)
    {
        return 0.f;
    }

    float3 SurfaceBSDFEval(float3 wi, float3 wo)
    {
        return {{SurfaceBSDF}}_Eval(wo, wi);
    }

    float SurfaceBSDFPdf(float3 wi, float3 wo)
    {
        return {{SurfaceBSDF}}_Pdf(wi, wo);
    }
    
    float3 SurfaceBSDFSamplef(float3 wo, float uc, float2 u, out float3 wi, out float pdf)
    {
        return {{SurfaceBSDF}}_Samplef(wo, uc, u, wi, pdf);
    }
};

#else

struct Material
{
    void Init()
    {
    }

    float3 SurfaceBSDFEval(float3 wi, float3 wo)
    {
        return 1.f;
    }

    float SurfaceBSDFPdf(float3 wi, float3 wo)
    {
        return 1.f;
    }
    
    float3 SurfaceBSDFSamplef(float3 wo, float uc, float2 u, out float3 wi, out float pdf)
    {
        return 1.f;
    }
};

#endif

#endif