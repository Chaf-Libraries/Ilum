#pragma once

namespace Ilum
{
class RHITexture;
class RHIBuffer;

static std::unique_ptr<RHITexture> MapTextureToCUDA(RHITexture *texture);
static std::unique_ptr<RHIBuffer>  MapBufferToCUDA(RHIBuffer *buffer);
}