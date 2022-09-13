#pragma once

#include "RHIDescriptor.hpp"

namespace Ilum::DX12
{
class Descriptor : public RHIDescriptor
{
  public:
	Descriptor(RHIDevice *device, const ShaderMeta &meta);

	~Descriptor() = default;

	virtual RHIDescriptor &BindTexture(const std::string &name, RHITexture *texture, RHITextureDimension dimension) override;
	virtual RHIDescriptor &BindTexture(const std::string &name, RHITexture *texture, const TextureRange &range) override;
	virtual RHIDescriptor &BindTexture(const std::string &name, const std::vector<RHITexture *> &textures, RHITextureDimension dimension) override;

	virtual RHIDescriptor &BindSampler(const std::string &name, RHISampler *sampler) override;
	virtual RHIDescriptor &BindSampler(const std::string &name, const std::vector<RHISampler *> &samplers) override;

	virtual RHIDescriptor &BindBuffer(const std::string &name, RHIBuffer *buffer) override;
	virtual RHIDescriptor &BindBuffer(const std::string &name, RHIBuffer *buffer, size_t offset, size_t range) override;
	virtual RHIDescriptor &BindBuffer(const std::string &name, const std::vector<RHIBuffer *> &buffers) override;

	virtual RHIDescriptor &BindConstant(const std::string &name, const void *constant) override;
};
}        // namespace Ilum::DX12