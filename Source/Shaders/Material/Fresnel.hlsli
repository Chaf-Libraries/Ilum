#ifndef FRESNEL_HLSLI
#define FRESNEL_HLSLI

#include "../Math.hlsli"

struct FresnelOp
{
    float3 Eval(float cos_theta_i)
    {
        return 1.f;
    }
};

float FresnelDielectric(float cos_theta_i, float eta)
{
   cos_theta_i = clamp(cos_theta_i, -1, 1);
   // Potentially flip interface orientation for Fresnel equations
   if (cos_theta_i < 0)
   {
       eta = 1 / eta;
       cos_theta_i = -cos_theta_i;
   }

   // Compute $\cos\,\theta_\roman{t}$ for Fresnel equations using Snell's law
   float sin2Theta_i = 1 - Sqr(cos_theta_i);
   float sin2Theta_t = sin2Theta_i / Sqr(eta);
   if (sin2Theta_t >= 1)
   {
       return 1.f;
   }
   float cos_theta_t = sqrt(1 - sin2Theta_t);

   float r_parl = (eta * cos_theta_i - cos_theta_t) / (eta * cos_theta_i + cos_theta_t);
   float r_perp = (cos_theta_i - eta * cos_theta_t) / (cos_theta_i + eta * cos_theta_t);
   return (Sqr(r_parl) + Sqr(r_perp)) / 2;
}

// struct FresnelDielectric
// {
//     float eta_i;
//     float eta_t;
    
//     float Eval(float cos_theta_i)
//     {
//         cos_theta_i = clamp(cos_theta_i, -1.0, 1.0);

// 	    // Potentially swap indices of refraction
//         if (cos_theta_i <= 0.f)
//         {
// 		// Swap
//             float temp = eta_i;
//             eta_i = eta_t;
//             eta_t = temp;
//             cos_theta_i = abs(cos_theta_i);
//         }

//         float sin_theta_i = sqrt(max(0.0, 1.0 - cos_theta_i * cos_theta_i));
//         float sin_theta_t = eta_i / eta_t * sin_theta_i;
//         if (sin_theta_t >= 1.0)
//         {
//             return 1.f;
//         }

//         float cos_theta_t = sqrt(max(0.0, 1.0 - sin_theta_t * sin_theta_t));
//         float Rparl = ((eta_t * cos_theta_i) - (eta_i * cos_theta_t)) /
// 	              ((eta_t * cos_theta_i) + (eta_i * cos_theta_t));
//         float Rperp = ((eta_i * cos_theta_i) - (eta_t * cos_theta_t)) /
// 	              ((eta_i * cos_theta_i) + (eta_t * cos_theta_t));

//         return (Rparl * Rparl + Rperp * Rperp) * 0.5f;
//     }
// };

float3 FresnelConductor(float cos_theta_i, float3 eta, float3 k)
{
    cos_theta_i = clamp(cos_theta_i, -1.0, 1.0);

    float cos_theta_i2 = cos_theta_i * cos_theta_i;
    float sin_theta_i2 = 1.0 - cos_theta_i2;
    float3 eta2 = eta * eta;
    float3 etak2 = k * k;

    float3 t0 = eta2 - etak2 - sin_theta_i2;
    float3 a2plusb2 = sqrt(t0 * t0 + 4.0 * eta2 * etak2);
    float3 t1 = a2plusb2 + cos_theta_i2;
    float3 a = sqrt(0.5 * (a2plusb2 + t0));
    float3 t2 = 2.0 * cos_theta_i * a;
    float3 Rs = (t1 - t2) / (t1 + t2);

    float3 t3 = cos_theta_i2 * a2plusb2 + sin_theta_i2 * sin_theta_i2;
    float3 t4 = t2 * sin_theta_i2;
    float3 Rp = Rs * (t3 - t4) / (t3 + t4);

    return 1;
    return 0.5 * (Rp + Rs);
}

//struct FresnelConductor
//{
//    float3 eta_i;
//    float3 eta_t;
//    float3 k;
//        
//    float3 Eval(float cos_theta_i)
//    {
//        cos_theta_i = clamp(cos_theta_i, -1.0, 1.0);
//
//        float3 eta = eta_t / eta_i;
//        float3 etak = k / eta_i;
//
//        float cos_theta_i2 = cos_theta_i * cos_theta_i;
//        float sin_theta_i2 = 1.0 - cos_theta_i2;
//        float3 eta2 = eta * eta;
//        float3 etak2 = etak * etak;
//
//        float3 t0 = eta2 - etak2 - sin_theta_i2;
//        float3 a2plusb2 = sqrt(t0 * t0 + 4.0 * eta2 * etak2);
//        float3 t1 = a2plusb2 + cos_theta_i2;
//        float3 a = sqrt(0.5 * (a2plusb2 + t0));
//        float3 t2 = 2.0 * cos_theta_i * a;
//        float3 Rs = (t1 - t2) / (t1 + t2);
//
//        float3 t3 = cos_theta_i2 * a2plusb2 + sin_theta_i2 * sin_theta_i2;
//        float3 t4 = t2 * sin_theta_i2;
//        float3 Rp = Rs * (t3 - t4) / (t3 + t4);
//
//        return 0.5 * (Rp + Rs);
//    }
//};

#endif