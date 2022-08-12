#pragma once

#include "RHI/RHIDescriptor.hpp"

#include <volk.h>

namespace Ilum::Vulkan
{
struct TextureResolve
{
	uint32_t    set     = 0;
	uint32_t    binding = 0;
	VkImageLayout layout  = VK_IMAGE_LAYOUT_UNDEFINED;

	std::vector<VkImageView> views;
	std::vector<VkSampler>   samplers;
};

struct BufferResolve
{
	uint32_t set = 0;
	uint32_t binding = 0;

	std::vector<VkBuffer> buffers;
	std::vector<size_t>   ranges;
	std::vector<size_t>   offsets;
};

struct AccelerationStructureResolve
{
	uint32_t set = 0;
	uint32_t binding = 0;

	std::vector<VkAccelerationStructureKHR> acceleration_structures;
};

class Descriptor : public RHIDescriptor
{
  public:
	Descriptor(RHIDevice *device, const ShaderMeta &meta);

	virtual ~Descriptor() override;

	virtual RHIDescriptor &BindTexture(const std::string &name, RHITexture *texture, RHITextureDimension dimension) override;
	virtual RHIDescriptor &BindTexture(const std::string &name, RHITexture *texture, RHITextureDimension dimension, uint32_t base_mip, uint32_t mip_count, uint32_t base_layer, uint32_t layer_count) override;
	virtual RHIDescriptor &BindTexture(const std::string &name, const std::vector<RHITexture *> &textures, RHITextureDimension dimension) override;

	virtual RHIDescriptor &BindSampler(const std::string &name, RHISampler *sampler) override;
	virtual RHIDescriptor &BindSampler(const std::string &name, const std::vector<RHISampler *> &samplers) override;

	virtual RHIDescriptor &BindBuffer(const std::string &name, RHIBuffer *buffer) override;
	virtual RHIDescriptor &BindBuffer(const std::string &name, RHIBuffer *buffer, size_t offset, size_t range) override;
	virtual RHIDescriptor &BindBuffer(const std::string &name, const std::vector<RHIBuffer *> &buffers) override;

	virtual RHIDescriptor &BindConstant(const std::string &name, const void *constant, size_t size) override;

	const std::unordered_map<uint32_t, VkDescriptorSet> &GetDescriptorSet();

	const std::unordered_map<uint32_t, VkDescriptorSetLayout> &GetDescriptorSetLayout();

  private:
	VkDescriptorSetLayout CreateDescriptorSetLayout(const ShaderMeta &meta);

  private:
	std::unordered_map<uint32_t, VkDescriptorSetLayout> m_descriptor_set_layouts;
	std::unordered_map<uint32_t, VkDescriptorSet>       m_descriptor_sets;

	std::unordered_map<std::string, std::pair<uint32_t, uint32_t>> m_descriptor_lookup;

	std::map<std::string, TextureResolve> m_texture_resolves;
	std::map<std::string, BufferResolve> m_buffer_resolve;
	std::map<std::string, AccelerationStructureResolve> m_acceleration_structures;

	std::unordered_map<std::string, size_t>                    m_binding_hash;

	std::unordered_map<uint32_t, bool> m_binding_dirty;

	std::unordered_map<std::string, std::pair<const void *, size_t>> m_constants;
};
}        // namespace Ilum::Vulkan