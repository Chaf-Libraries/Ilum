#include "RHITexture.hpp"
#include "RHIDevice.hpp"

#include <Core/Plugin.hpp>

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

const std::string RHITexture::GetBackend() const
{
	return p_device->GetBackend();
}

std::unique_ptr<RHITexture> RHITexture::Alias(const TextureDesc &desc)
{
	return nullptr;
}

std::unique_ptr<RHITexture> RHITexture::Create(RHIDevice *device, const TextureDesc &desc)
{
	return std::unique_ptr<RHITexture>(std::move(PluginManager::GetInstance().Call<RHITexture *>(fmt::format("shared/RHI/RHI.{}.dll", device->GetBackend()), "CreateTexture", device, desc)));
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