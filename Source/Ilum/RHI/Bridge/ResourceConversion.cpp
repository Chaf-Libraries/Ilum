#include "ResourceConversion.hpp"
#include "RHIBuffer.hpp"
#include "RHITexture.hpp"

namespace Ilum
{
inline std::unique_ptr<RHITexture> MapTextureVulkanToCUDA(RHITexture *texture)
{


	return nullptr;
}

inline std::unique_ptr<RHIBuffer> MapBufferVulkanToCUDA(RHIBuffer *buffer)
{
	return nullptr;
}

std::unique_ptr<RHITexture> MapTextureToCUDA(RHITexture *texture)
{
	switch (texture->GetBackend())
	{
		case RHIBackend::Vulkan:
			MapTextureVulkanToCUDA(texture);
			break;
		default:
			break;
	}
	return nullptr;
}

std::unique_ptr<RHIBuffer> MapBufferToCUDA(RHIBuffer *buffer)
{
	switch (buffer->GetBackend())
	{
		case RHIBackend::Vulkan:
			MapBufferVulkanToCUDA(buffer);
		default:
			break;
	}
	return nullptr;
}
}        // namespace Ilum