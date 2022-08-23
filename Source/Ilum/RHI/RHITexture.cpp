#include "RHITexture.hpp"

#ifdef RHI_BACKEND_VULKAN
#	include "Backend/Vulkan/Texture.hpp"
#elif defined RHI_BACKEND_DX12
#	include "Backend/DX12/Texture.hpp"
#endif        // RHI_BACKEND

namespace Ilum
{
RHITexture::RHITexture(RHIDevice *device, const TextureDesc &desc) :
    p_device(device), m_desc(desc)
{
}

const TextureDesc &RHITexture::GetDesc() const
{
	return m_desc;
}

std::unique_ptr<RHITexture> RHITexture::Alias(const TextureDesc &desc)
{
	return Create(p_device, desc);
}

std::unique_ptr<RHITexture> RHITexture::Create(RHIDevice *device, const TextureDesc &desc)
{
#ifdef RHI_BACKEND_VULKAN
	return std::make_unique<Vulkan::Texture>(device, desc);
#elif defined RHI_BACKEND_DX12
	return std::make_unique<DX12::Texture>(device, desc);
#endif
	return nullptr;
}

std::unique_ptr<RHITexture> RHITexture::Create2D(RHIDevice *device, uint32_t width, uint32_t height, RHIFormat format, RHITextureUsage usage, bool mipmap, uint32_t samples)
{
	TextureDesc desc = {};
	desc.width       = width;
	desc.height      = height;
	desc.depth       = 1;
	desc.layers      = 1;
	desc.mips        = mipmap ? static_cast<uint32_t>(std::floor(std::log2(std::max(width, height)))) + 1 : 1;
	desc.samples     = samples;
	desc.format      = format;
	desc.usage       = usage;

	return Create(device, desc);
}

std::unique_ptr<RHITexture> RHITexture::Create3D(RHIDevice *device, uint32_t width, uint32_t height, uint32_t depth, RHIFormat format, RHITextureUsage usage)
{
	TextureDesc desc = {};
	desc.width       = width;
	desc.height      = height;
	desc.depth       = depth;
	desc.layers      = 1;
	desc.mips        = 1;
	desc.samples     = 1;
	desc.format      = format;
	desc.usage       = usage;

	return Create(device, desc);
}

std::unique_ptr<RHITexture> RHITexture::CreateCube(RHIDevice *device, uint32_t width, uint32_t height, RHIFormat format, RHITextureUsage usage, bool mipmap)
{
	TextureDesc desc = {};
	desc.width       = width;
	desc.height      = height;
	desc.depth       = 1;
	desc.layers      = 6;
	desc.mips        = mipmap ? static_cast<uint32_t>(std::floor(std::log2(std::max(width, height)))) + 1 : 1;
	desc.samples     = 1;
	desc.format      = format;
	desc.usage       = usage;

	return Create(device, desc);
}

std::unique_ptr<RHITexture> RHITexture::Create2DArray(RHIDevice *device, uint32_t width, uint32_t height, uint32_t layers, RHIFormat format, RHITextureUsage usage, bool mipmap, uint32_t samples)
{
	TextureDesc desc = {};
	desc.width       = width;
	desc.height      = height;
	desc.depth       = 1;
	desc.layers      = layers;
	desc.mips        = mipmap ? static_cast<uint32_t>(std::floor(std::log2(std::max(width, height)))) + 1 : 1;
	desc.samples     = samples;
	desc.format      = format;
	desc.usage       = usage;

	return Create(device, desc);
}
}        // namespace Ilum