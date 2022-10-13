struct {{BxDFName}}
{
    static const uint BxDF_Type = {{BxDFTypeA}}::BxDF_Type | {{BxDFTypeB}}::BxDF_Type;
    {{BxDFTypeA}} bxdf_A;
    {{BxDFTypeB}} bxdf_B;
    float weight;
    
    static {{BxDFName}} Create()
    {
        {{#Definitions}}
        {{Definition}}
        {{/Definitions}}
        {{BxDFName}} result;
        result.bxdf_A = {{BxDFTypeA}}::Create();
        result.bxdf_B = {{BxDFTypeB}}::Create();
        result.weight = {{&Weight}};
        return result;
    }
     
    float3 Eval(float3 wi, float3 wo)
    {
        return bxdf_A.Eval(wi, wo) * (1.0 - weight) + bxdf_B.Eval(wi, wo) * weight;
    }
    
    float Pdf(float3 wi, float3 wo)
    {
        return bxdf_A.Pdf(wi, wo) * (1.0 - weight) + bxdf_B.Pdf(wi, wo) * weight;
    }
    
    float3 Samplef(float3 wi, float sample1, float2 sample2, inout float3 wo, inout float pdf)
    {
        return 0.f;
    }
};