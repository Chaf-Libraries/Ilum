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

	virtual ~RHIDescriptor() = 0;

	virtual RHIDescriptor &BindTextureUAV(const std::string &name, RHITexture *texture)                            = 0;
	virtual RHIDescriptor &BindTextureUAV(const std::string &name, RHITexture *texture, const TextureRange &range) = 0;
	virtual RHIDescriptor &BindTextureUAV(const std::string &name, const std::vector<RHITexture *> &textures)      = 0;

	virtual RHIDescriptor &BindTextureSRV(const std::string &name, RHITexture *texture)                            = 0;
	virtual RHIDescriptor &BindTextureSRV(const std::string &name, RHITexture *texture, const TextureRange &range) = 0;
	virtual RHIDescriptor &BindTextureSRV(const std::string &name, const std::vector<RHITexture *> &textures)      = 0;

	virtual RHIDescriptor &BindSampler(const std::string &name, RHISampler *sampler)                      = 0;
	virtual RHIDescriptor &BindSampler(const std::string &name, const std::vector<RHISampler *> &sampler) = 0;

	virtual RHIDescriptor &BindBuffer(const std::string &name, RHIBuffer *buffer)                      = 0;
	virtual RHIDescriptor &BindBuffer(const std::string &name, const std::vector<RHIBuffer *> &buffer) = 0;

	virtual RHIDescriptor &BindConstant(const std::string &name, void *constant, size_t size) = 0;

	// RHIDescriptor &BindAccelerationStructure();

  private:
	RHIDevice *p_device = nullptr;
};
}        // namespace Ilum