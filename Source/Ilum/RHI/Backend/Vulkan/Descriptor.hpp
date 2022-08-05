#pragma once

#include "RHI/RHIDescriptor.hpp"

namespace Ilum::Vulkan
{
class Descriptor : public RHIDescriptor
{
  public:
	Descriptor(RHIDevice *device, const ShaderMeta &meta);

	virtual ~Descriptor() override;

	virtual RHIDescriptor &BindTextureUAV(const std::string &name, RHITexture *texture) override;
	virtual RHIDescriptor &BindTextureUAV(const std::string &name, RHITexture *texture, const TextureRange &range) override;
	virtual RHIDescriptor &BindTextureUAV(const std::string &name, const std::vector<RHITexture *> &textures) override;

	virtual RHIDescriptor &BindTextureSRV(const std::string &name, RHITexture *texture) override;
	virtual RHIDescriptor &BindTextureSRV(const std::string &name, RHITexture *texture, const TextureRange &range) override;
	virtual RHIDescriptor &BindTextureSRV(const std::string &name, const std::vector<RHITexture *> &textures) override;

	virtual RHIDescriptor &BindSampler(const std::string &name, RHISampler *sampler) override;
	virtual RHIDescriptor &BindSampler(const std::string &name, const std::vector<RHISampler *> &sampler) override;

	virtual RHIDescriptor &BindBuffer(const std::string &name, RHIBuffer *buffer) override;
	virtual RHIDescriptor &BindBuffer(const std::string &name, const std::vector<RHIBuffer *> &buffer) override;

	virtual RHIDescriptor &BindConstant(const std::string &name, void *constant, size_t size) override;

  private:
	std::unordered_map<std::string, uint32_t> m_binding_table;


};
}        // namespace Ilum::Vulkan