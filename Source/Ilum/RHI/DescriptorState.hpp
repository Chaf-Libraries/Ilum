#pragma once

#include "AccelerateStructure.hpp"
#include "Buffer.hpp"
#include "Sampler.hpp"
#include "ShaderReflection.hpp"
#include "Texture.hpp"

#include <map>
#include <vector>

namespace Ilum
{
class RHIDevice;
class PipelineState;

class DescriptorState
{
	friend class CommandBuffer;

  public:
	DescriptorState(RHIDevice *device, const PipelineState *pso);
	~DescriptorState() = default;

	DescriptorState &Bind(uint32_t set, uint32_t binding, Buffer *buffer);
	DescriptorState &Bind(uint32_t set, uint32_t binding, VkImageView view, VkSampler sampler = VK_NULL_HANDLE);
	DescriptorState &Bind(uint32_t set, uint32_t binding, VkSampler sampler);
	DescriptorState &Bind(uint32_t set, uint32_t binding, AccelerationStructure *acceleration_structure);
	DescriptorState &Bind(uint32_t set, uint32_t binding, const std::vector<Buffer *> &buffers);
	DescriptorState &Bind(uint32_t set, uint32_t binding, const std::vector<VkImageView> &views, VkSampler sampler = VK_NULL_HANDLE);
	DescriptorState &Bind(uint32_t set, uint32_t binding, const std::vector<VkSampler> &samplers);
	DescriptorState &Bind(uint32_t set, uint32_t binding, const std::vector<AccelerationStructure *> &acceleration_structures);

	void Write();

  private:
	RHIDevice *p_device = nullptr;
	const PipelineState *p_pso = nullptr;

	std::map<uint32_t, std::map<uint32_t, std::vector<VkDescriptorBufferInfo>>>  m_buffer_resolves;
	std::map<uint32_t, std::map<uint32_t, std::vector<VkDescriptorImageInfo>>>   m_image_resolves;
	std::map<uint32_t, std::map<uint32_t, std::vector<AccelerationStructure *>>> m_acceleration_structure_resolves;
	std::map<uint32_t, std::map<uint32_t, VkWriteDescriptorSet>>                 m_resolves;

	bool m_dirty = false;

	std::map<uint32_t, VkDescriptorSet> m_descriptor_sets;

	VkPipelineBindPoint m_bind_point = VK_PIPELINE_BIND_POINT_MAX_ENUM;
};

}        // namespace Ilum