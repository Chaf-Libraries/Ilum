struct {{BxDFName}}
{
    static const uint BxDF_Type = {{BxDFType}}::BxDF_Type;
    {{BxDFType}} bxdf;
    
    static {{BxDFName}} Create()
    {
        {{BxDFName}} result;
        {{&Definitions}}
        result.bxdf = {{BxDFType}}::Create({{Parameter}});
        return result;
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
        return bxdf.Samplef(wi, sample1, sample2, wo, pdf);
    }
};