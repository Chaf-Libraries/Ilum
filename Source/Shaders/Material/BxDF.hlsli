float3 {{BxDFName}}_Eval(float3 wo, float3 wi)
{
    return {{EvalFunc}};
}
    
float {{BxDFName}}_Pdf(float3 wo, float3 wi)
{
    return {{PdfFunc}};
}
    
float3 {{BxDFName}}_Samplef(float3 wo, float uc, float2 u, out float3 wi, out float pdf)
{
    return {{SamplefFunc}};
}
