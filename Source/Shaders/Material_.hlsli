#ifndef MATERIAL_HLSLI
#define MATERIAL_HLSLI

#include "Attribute.hlsli"
#include "Random.hlsli"

#include "Material/BSDF/BSDF.hlsli"
{{#Headers}}
{{&Header}}
{{/Headers}}

Texture2D<float4> Textures[];

cbuffer MaterialBuffer
{
    {{#Textures}}
    uint {{Texture}};
    {{/Textures}}
}

{{#Samplers}}
SamplerState {{Sampler}};
{{/Samplers}}

struct BSDF
{
    {{#Declarations}}
    {{Declaration}}
    {{/Declarations}}
    float3 ns, ng;
    float3 ss, ts;

    void Init()
    {
        {{#Initializations}}
        {{Initialization}}
        {{/Initializations}}
    }

    float3 WorldToLocal(float3 v) 
    {
        return float3(dot(v, ss), dot(v, ts), dot(v, ns));
    }

    float3 LocalToWorld(float3 v) 
    {
        return float3(ss.x * v.x + ts.x * v.y + ns.x * v.z,
                        ss.y * v.x + ts.y * v.y + ns.y * v.z,
                        ss.z * v.x + ts.z * v.y + ns.z * v.z);
    }

    float3 Eval(float3 woW, float3 wiW, uint bxdf_type)
    {
        float3 wi = WorldToLocal(wiW);
        float3 wo = WorldToLocal(woW);

        if (wo.z == 0)
        {
            return 0.f;
        }

        bool reflect = dot(wiW, ng) * dot(woW, ng) > 0;
        float3 f = float3(0.f, 0.f, 0.f);
        {{#EvaluateBSDF}}
        if ({{BxDF}}.MatchesFlags(bxdf_type) &&
            ((reflect && ({{BxDF}}.flags & BSDF_REFLECTION)) ||
            (!reflect && ({{BxDF}}.flags & BSDF_TRANSMISSION))))
        {
            f += {{BxDF}}.Eval(wo, wi) * {{Weight}};
        }
        {{/EvaluateBSDF}}

        return f;
    }

    float Pdf(float3 woW, float3 wiW, uint bxdf_type)
    {
        float3 wo = WorldToLocal(woW);
        float3 wi = WorldToLocal(wiW);
        if(wo.z == 0)
        {
            return 0.f;
        }

        float pdf = 0.f;
        int mathching_comps = 0;

        {{#CalculatePDF}}
        if ({{BxDF}}.MatchesFlags(bxdf_type)) 
        {
            ++mathching_comps;
            pdf += {{BxDF}}.Pdf(wo, wi);
        }
        {{/CalculatePDF}}

        return mathching_comps > 0 ? pdf / mathching_comps : 0.f;
    }
    
    float3 Samplef(float3 woW, float2 u, out float3 wiW, out float pdf, uint bxdf_type)
    {
        int matching_comps = 0;

        {{#MatchingComponents}}
        if ({{BxDF}}.MatchesFlags(bxdf_type)) 
        {
            ++matching_comps;
        }
        {{/MatchingComponents}}

        if (matching_comps == 0) 
        {
            pdf = 0;
            return float3(0.f, 0.f, 0.f);
        }

        int comp = min((int)(floor(u.x * float(matching_comps))), matching_comps - 1);

        int count = comp;
        bool found = false;

        float3 f = float3(0.f, 0.f, 0.f);

        float3 wi = float3(0.f, 0.f, 0.f);
        float3 wo = WorldToLocal(woW);

        uint sample_type = 0;

        {{#SampleBSDF}}
        if(!found && {{BxDF}}.MatchFlags(bxdf_type) && count-- == 0)
        {
            sample_type = {{BxDF}}.flags;
            f = {{BxDF}}.Samplef(wo, u, wi, pdf) * {{Weight}};
            wiW = LocalToWorld(wi);
            found = true;
            if(pdf == 0)
            {
                return float3(0.f, 0.f, 0.f);
            }
        }
        {{/SampleBSDF}}

        if (!(sample_type & BSDF_SPECULAR) && matching_comps > 1)
        {
            pdf = Pdf(woW, wiW, bxdf_type);
        }

        if (!(sample_type & BSDF_SPECULAR))
        {
            f = Eval(woW, wiW, sample_type)
        }

        return f;
    }
}

struct Material
{
    BSDF bsdf;

    void Init()
    {
        bsdf.Init();
    }
}

#endif