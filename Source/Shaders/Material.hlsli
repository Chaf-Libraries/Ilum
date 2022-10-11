#ifndef MATERIAL_HLSLI
#define MATERIAL_HLSLI

{{#Include}}
{{&IncludePath}}
{{/Include}}

cbuffer MaterialBuffer
{
    {{#Resource}}
    {{TypeName}} {{VariableName}};
    {{/Resource}}
};

{{#Definitions}}

{{/Definitions}}


struct Material
{
    {{BxDFType}} bxdf;

    void Init()
    {
        bxdf.Init();
    }

    float3 Eval(float3 wi, float3 wo)
    {
        return bxdf.Eval(wi, wo);
    }
    
    float Pdf(float3 wi, float3 wo)
    {
        return bxdf.Pdf(wi, wo);
    }
    
    float3 Samplef(float3 wi, float sample1, float2 sample2, out float3 wo, out float pdf)
    {
        return bxdf.Samplef(wi, sample1, sample2. wo, pdf);
    }
};

#endif