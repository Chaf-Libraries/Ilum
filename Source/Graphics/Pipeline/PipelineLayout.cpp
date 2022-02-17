#include "PipelineLayout.hpp"
#include "../Device/Device.hpp"
#include "../Shader/SpirvReflection.hpp"

namespace Ilum::Graphics
{
PipelineLayout::PipelineLayout(const Device &device, const ReflectionData &reflection_data, const std::vector<VkDescriptorSetLayout> &descriptor_set_layouts) :
    m_device(device)
{
	std::vector<VkPushConstantRange> push_constant_ranges;
	for (auto &constant : reflection_data.constants)
	{
		if (constant.type == ReflectionData::Constant::Type::Push)
		{
			VkPushConstantRange push_constant_range = {};
			push_constant_range.stageFlags          = constant.stage;
			push_constant_range.size                = constant.size;
			push_constant_range.offset              = constant.offset;
			push_constant_ranges.push_back(push_constant_range);
		}
	}

	VkPipelineLayoutCreateInfo pipeline_layout_create_info = {};
	pipeline_layout_create_info.sType                      = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
	pipeline_layout_create_info.pushConstantRangeCount     = static_cast<uint32_t>(push_constant_ranges.size());
	pipeline_layout_create_info.pPushConstantRanges        = push_constant_ranges.data();
	pipeline_layout_create_info.setLayoutCount             = static_cast<uint32_t>(descriptor_set_layouts.size());
	pipeline_layout_create_info.pSetLayouts                = descriptor_set_layouts.data();

	vkCreatePipelineLayout(m_device, &pipeline_layout_create_info, nullptr, &m_handle);
}

PipelineLayout::~PipelineLayout()
{
	if (m_handle)
	{
		vkDestroyPipelineLayout(m_device, m_handle, nullptr);
	}
}

PipelineLayout::operator const VkPipelineLayout &() const
{
	return m_handle;
}

const VkPipelineLayout &PipelineLayout::GetHandle() const
{
	return m_handle;
}
}        // namespace Ilum::Graphics