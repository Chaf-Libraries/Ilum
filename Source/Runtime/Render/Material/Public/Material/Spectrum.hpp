#pragma once

#include <Core/Core.hpp>

#include <glm/glm.hpp>

namespace Ilum
{
static const int sampledLambdaStart = 400;
static const int sampledLambdaEnd   = 700;
static const int nSpectralSamples   = 60;

extern void XYZToRGB(const float xyz[3], float rgb[3]);
extern void RGBToXYZ(const float rgb[3], float xyz[3]);

static const int   nCIESamples = 471;
extern const float CIE_X[nCIESamples];
extern const float CIE_Y[nCIESamples];
extern const float CIE_Z[nCIESamples];
extern const float CIE_lambda[nCIESamples];
static const float CIE_Y_integral    = 106.856895f;

extern bool  SpectrumSamplesSorted(const float *lambda, const float *vals, int32_t n);
extern void  SortSpectrumSamples(float *lambda, float *vals, int32_t n);
extern float InterpolateSpectrumSamples(const float *lambda, const float *vals, int32_t n, float l);

glm::vec3 FromSampled(const float *lambda, const float *v, int32_t n);
}        // namespace Ilum