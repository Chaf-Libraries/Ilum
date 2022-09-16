#pragma once

#include "RHI/RHIDescriptor.hpp"

#include <volk.h>

namespace Ilum::Vulkan
{
struct TextureResolve
{
	uint32_t      set     = 0;
	uint32_t      binding = 0;
	VkImageLayout layout  = VK_IMAGE_LAYOUT_UNDEFINED;

	std::vector<VkImageView> views;
	std::vector<VkSampler>   samplers;
};

struct BufferResolve
{
	uint32_t set     = 0;
	uint32_t binding = 0;

	std::vector<VkBuffer> buffers;
	std::vector<size_t>   ranges;
	std::vector<size_t>   offsets;
};

struct AccelerationStructureResolve
{
	uint32_t set     = 0;
	uint32_t binding = 0;

	std::vector<VkAccelerationStructureKHR> acceleration_structures;
};

struct ConstantResolve
{
	std::vector<uint8_t> data;
	size_t             offset = 0;
	VkShaderStageFlags stage;
};

class Descriptor : public RHIDescriptor
{
  public:
	Descriptor(RHIDevice *device, const ShaderMeta &meta);

	virtual ~Descriptor() override;

	virtual RHIDescriptor &BindTexture(const std::string &name, RHITexture *texture, RHITextureDimension dimension) override;
	virtual RHIDescriptor &BindTexture(const std::string &name, RHITexture *texture, const TextureRange &range) override;
	virtual RHIDescriptor &BindTexture(const std::string &name, const std::vector<RHITexture *> &textures, RHITextureDimension dimension) override;

	virtual RHIDescriptor &BindSampler(const std::string &name, RHISampler *sampler) override;
	virtual RHIDescriptor &BindSampler(const std::string &name, const std::vector<RHISampler *> &samplers) override;

	virtual RHIDescriptor &BindBuffer(const std::string &name, RHIBuffer *buffer) override;
	virtual RHIDescriptor &BindBuffer(const std::string &name, RHIBuffer *buffer, size_t offset, size_t range) override;
	virtual RHIDescriptor &BindBuffer(const std::string &name, const std::vector<RHIBuffer *> &buffers) override;

	virtual RHIDescriptor &BindConstant(const std::string &name, const void *constant) override;

	virtual RHIDescriptor &BindAccelerationStructure(const std::string &name, RHIAccelerationStructure *acceleration_structure) override;

	const std::unordered_map<uint32_t, VkDescriptorSet> &GetDescriptorSet();

	const std::unordered_map<uint32_t, VkDescriptorSetLayout> &GetDescriptorSetLayout();

	const std::map<std::string, ConstantResolve> &GetConstantResolve() const;

  private:
	VkDescriptorSetLayout CreateDescriptorSetLayout(const ShaderMeta &meta);

  private:
	std::unordered_map<uint32_t, VkDescriptorSetLayout> m_descriptor_set_layouts;
	std::unordered_map<uint32_t, VkDescriptorSet>       m_descriptor_sets;

	std::unordered_map<std::string, std::pair<uint32_t, uint32_t>> m_descriptor_lookup;

	std::map<std::string, TextureResolve>               m_texture_resolves;
	std::map<std::string, BufferResolve>                m_buffer_resolves;
	std::map<std::string, AccelerationStructureResolve> m_acceleration_structure_resolves;
	std::map<std::string, ConstantResolve>              m_constant_resolves;

	std::unordered_map<std::string, size_t> m_binding_hash;

	std::unordered_map<uint32_t, bool> m_binding_dirty;
};
}        // namespace Ilum::Vulkan