#include "Descriptor.hpp"

namespace Ilum::Vulkan
{
Descriptor::Descriptor(RHIDevice *device, const ShaderMeta &meta) :
    RHIDescriptor(device, meta)
{
}

Descriptor ::~Descriptor()
{
}

RHIDescriptor &Descriptor::BindTextureUAV(const std::string &name, RHITexture *texture)
{
	return *this;
}

RHIDescriptor &Descriptor::BindTextureUAV(const std::string &name, RHITexture *texture, const TextureRange &range)
{
	return *this;
}

RHIDescriptor &Descriptor::BindTextureUAV(const std::string &name, const std::vector<RHITexture *> &textures)
{
	return *this;
}

RHIDescriptor &Descriptor::BindTextureSRV(const std::string &name, RHITexture *texture)
{
	return *this;
}

RHIDescriptor &Descriptor::BindTextureSRV(const std::string &name, RHITexture *texture, const TextureRange &range)
{
	return *this;
}

RHIDescriptor &Descriptor::BindTextureSRV(const std::string &name, const std::vector<RHITexture *> &textures)
{
	return *this;
}

RHIDescriptor &Descriptor::BindSampler(const std::string &name, RHISampler *sampler)
{
	return *this;
}

RHIDescriptor &Descriptor::BindSampler(const std::string &name, const std::vector<RHISampler *> &sampler)
{
	return *this;
}

RHIDescriptor &Descriptor::BindBuffer(const std::string &name, RHIBuffer *buffer)
{
	return *this;
}

RHIDescriptor &Descriptor::BindBuffer(const std::string &name, const std::vector<RHIBuffer *> &buffer)
{
	return *this;
}

RHIDescriptor &Descriptor::BindConstant(const std::string &name, void *constant, size_t size)
{
	return *this;
}
}        // namespace Ilum::Vulkan