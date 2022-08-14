#include "PipelineState.hpp"
#include "Descriptor.hpp"
#include "Device.hpp"

#include <volk.h>

namespace Ilum::Vulkan
{
static VkPipelineCache                              PipelineCache;
static std::unordered_map<size_t, VkPipeline>       Pipelines;
static std::unordered_map<size_t, VkPipelineLayout> PipelineLayouts;

static uint32_t PipelineCount = 0;

PipelineState::PipelineState(RHIDevice *device, Descriptor *descriptor) :
    RHIPipelineState(device)
{
	if (PipelineCount++ == 0)
	{
		if (!PipelineCache)
		{
			VkPipelineCacheCreateInfo create_info = {};
			create_info.sType                     = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
			vkCreatePipelineCache(static_cast<Device *>(p_device)->GetDevice(), &create_info, nullptr, &PipelineCache);
		}
	}

	const auto &meta = descriptor->GetShaderMeta();

	size_t hash = 0;
	HashCombine(hash, meta.hash, m_hash);

	// Create pipeline layout
	if (PipelineLayouts.find(hash) != PipelineLayouts.end())
	{
		m_pipeline_layout = PipelineLayouts[hash];
	}
	else
	{
		std::vector<VkPushConstantRange> push_constants;
		for (auto &constant : meta.constants)
		{
			VkPushConstantRange push_constant_range = {};
			push_constant_range.stageFlags          = ToVulkanShaderStage(constant.stage);
			push_constant_range.size                = constant.size;
			push_constant_range.offset              = constant.offset;
			push_constants.push_back(push_constant_range);
		}

		std::vector<VkDescriptorSetLayout> descriptor_set_layouts;
		for (auto &[set, layout] : descriptor->GetDescriptorSetLayout())
		{
			descriptor_set_layouts.push_back(layout);
		}

		VkPipelineLayoutCreateInfo pipeline_layout_create_info = {};
		pipeline_layout_create_info.sType                      = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		pipeline_layout_create_info.pushConstantRangeCount     = static_cast<uint32_t>(push_constants.size());
		pipeline_layout_create_info.pPushConstantRanges        = push_constants.data();
		pipeline_layout_create_info.setLayoutCount             = static_cast<uint32_t>(descriptor_set_layouts.size());
		pipeline_layout_create_info.pSetLayouts                = descriptor_set_layouts.data();

		vkCreatePipelineLayout(static_cast<Device *>(p_device)->GetDevice(), &pipeline_layout_create_info, nullptr, &m_pipeline_layout);

		PipelineLayouts.emplace(hash, m_pipeline_layout);
	}

	// Create pipeline
	if (Pipelines.find(hash) != Pipelines.end())
	{
		m_pipeline = Pipelines[hash];
	}
	else
	{

	}
}

PipelineState ::~PipelineState()
{
	m_pipeline        = VK_NULL_HANDLE;
	m_pipeline_layout = VK_NULL_HANDLE;

	if (--PipelineCount == 0)
	{
		for (auto &[hash, pipeline] : Pipelines)
		{
			vkDestroyPipeline(static_cast<Device *>(p_device)->GetDevice(), pipeline, nullptr);
		}
		for (auto &[hash, layout] : PipelineLayouts)
		{
			vkDestroyPipelineLayout(static_cast<Device *>(p_device)->GetDevice(), layout, nullptr);
		}
		Pipelines.clear();
		PipelineLayouts.clear();

		if (PipelineCache)
		{
			vkDestroyPipelineCache(static_cast<Device *>(p_device)->GetDevice(), PipelineCache, nullptr);
			PipelineCache = VK_NULL_HANDLE;
		}
	}
}

VkPipelineLayout PipelineState::GetPipelineLayout() const
{
	return m_pipeline_layout;
}

VkPipeline PipelineState::GetPipeline() const
{
	return m_pipeline;
}

}        // namespace Ilum::Vulkan