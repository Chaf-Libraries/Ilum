#ifndef BXDFTYPE_HLSLI
#define BXDFTYPE_HLSLI

static const uint BxDF_REFLECTION = 1 << 0;
static const uint BxDF_TRANSMISSION = 1 << 1;
static const uint BxDF_DIFFUSE = 1 << 2;
static const uint BxDF_GLOSSY = 1 << 3;
static const uint BxDF_SPECULAR = 1 << 4;
static const uint BxDF_ALL = BxDF_DIFFUSE | BxDF_GLOSSY | BxDF_SPECULAR | BxDF_REFLECTION | BxDF_TRANSMISSION;

#endif