#pragma once

#include "RHIBuffer.hpp"
#include "RHISampler.hpp"
#include "RHIShader.hpp"
#include "RHITexture.hpp"

#include <string>
#include <vector>

namespace Ilum
{
class RHIDevice;

class RHIDescriptor
{
  public:
	RHIDescriptor(RHIDevice *device, const ShaderMeta &meta);

	virtual ~RHIDescriptor() = default;

	const ShaderMeta &GetShaderMeta() const;

	static std::unique_ptr<RHIDescriptor> Create(RHIDevice *device, const ShaderMeta &meta);

	virtual RHIDescriptor &BindTexture(const std::string &name, RHITexture *texture, RHITextureDimension dimension)                       = 0;
	virtual RHIDescriptor &BindTexture(const std::string &name, RHITexture *texture, const TextureRange &range)                           = 0;
	virtual RHIDescriptor &BindTexture(const std::string &name, const std::vector<RHITexture *> &textures, RHITextureDimension dimension) = 0;

	virtual RHIDescriptor &BindSampler(const std::string &name, RHISampler *sampler)                       = 0;
	virtual RHIDescriptor &BindSampler(const std::string &name, const std::vector<RHISampler *> &samplers) = 0;

	virtual RHIDescriptor &BindBuffer(const std::string &name, RHIBuffer *buffer)                              = 0;
	virtual RHIDescriptor &BindBuffer(const std::string &name, RHIBuffer *buffer, size_t offset, size_t range) = 0;
	virtual RHIDescriptor &BindBuffer(const std::string &name, const std::vector<RHIBuffer *> &buffers)        = 0;

	virtual RHIDescriptor &BindConstant(const std::string &name, const void *constant) = 0;

	// RHIDescriptor &BindAccelerationStructure();
  protected:
	RHIDevice       *p_device = nullptr;
	const ShaderMeta m_meta;
};
}        // namespace Ilum