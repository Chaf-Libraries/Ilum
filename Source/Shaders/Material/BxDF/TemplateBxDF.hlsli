struct {{TemplateBxDFs}}
{
    {{BxDFType}} {{BxDFName}};

    void Init()
    {
        {{BxDFName}}.Init({{InitParameter}});
    }

    float3 Eval(float3 wi, float3 wo)
    {
        return {{BxDFName}}.Eval(wi, wo);
    }
    
    float Pdf(float3 wi, float3 wo)
    {
        return {{BxDFName}}.Pdf(wi, wo);
    }
    
    float3 Samplef(float3 wi, float sample1, float2 sample2, out float3 wo, out float pdf)
    {
        return {{BxDFName}}.Samplef(wi, sample1, sample2, wo, pdf);
    }
}