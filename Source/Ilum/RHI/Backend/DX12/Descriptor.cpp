#include "Descriptor.hpp"

namespace Ilum::DX12
{
Descriptor::Descriptor(RHIDevice *device, const ShaderMeta &meta) :
    RHIDescriptor(device, meta)
{
}

RHIDescriptor &Descriptor::BindTexture(const std::string &name, RHITexture *texture, RHITextureDimension dimension)
{
	return *this;
}

RHIDescriptor &Descriptor::BindTexture(const std::string &name, RHITexture *texture, const TextureRange &range)
{
	return *this;
}

RHIDescriptor &Descriptor::BindTexture(const std::string &name, const std::vector<RHITexture *> &textures, RHITextureDimension dimension)
{
	return *this;
}

RHIDescriptor &Descriptor::BindSampler(const std::string &name, RHISampler *sampler)
{
	return *this;
}

RHIDescriptor &Descriptor::BindSampler(const std::string &name, const std::vector<RHISampler *> &samplers)
{
	return *this;
}

RHIDescriptor &Descriptor::BindBuffer(const std::string &name, RHIBuffer *buffer)
{
	return *this;
}

RHIDescriptor &Descriptor::BindBuffer(const std::string &name, RHIBuffer *buffer, size_t offset, size_t range)
{
	return *this;
}

RHIDescriptor &Descriptor::BindBuffer(const std::string &name, const std::vector<RHIBuffer *> &buffers)
{
	return *this;
}

RHIDescriptor &Descriptor::BindConstant(const std::string &name, const void *constant)
{
	return *this;
}
}        // namespace Ilum::DX12