#include "PipelineState.hpp"
#include "Definitions.hpp"
#include "Descriptor.hpp"
#include "Device.hpp"
#include "RenderTarget.hpp"
#include "Shader.hpp"

#include <volk.h>

namespace Ilum::Vulkan
{
class ShaderBindingTableInfo
{
  public:
	ShaderBindingTableInfo(Device *device, uint32_t handle_count) :
	    p_device(device)
	{
		if (handle_count == 0)
		{
			return;
		}

		VkPhysicalDeviceRayTracingPipelinePropertiesKHR raytracing_pipeline_properties = {};
		raytracing_pipeline_properties.sType                                           = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR;
		VkPhysicalDeviceProperties2 deviceProperties2                                  = {};
		deviceProperties2.sType                                                        = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
		deviceProperties2.pNext                                                        = &raytracing_pipeline_properties;
		vkGetPhysicalDeviceProperties2(p_device->GetPhysicalDevice(), &deviceProperties2);

		uint32_t handle_size_aligned = (raytracing_pipeline_properties.shaderGroupHandleSize + raytracing_pipeline_properties.shaderGroupHandleAlignment - 1) &
		                               ~(raytracing_pipeline_properties.shaderGroupHandleAlignment - 1);

		VkBufferCreateInfo buffer_info = {};
		buffer_info.sType              = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
		buffer_info.usage              = VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
		buffer_info.size               = static_cast<size_t>(handle_count) * raytracing_pipeline_properties.shaderGroupHandleSize;

		VmaAllocationCreateInfo memory_info{};
		memory_info.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT;
		memory_info.usage = VMA_MEMORY_USAGE_CPU_TO_GPU;

		VmaAllocationInfo allocation_info{};
		vmaCreateBuffer(p_device->GetAllocator(),
		                &buffer_info, &memory_info,
		                &m_buffer, &m_allocation,
		                &allocation_info);

		m_memory = allocation_info.deviceMemory;

		VkBufferDeviceAddressInfoKHR buffer_device_address_info{};
		buffer_device_address_info.sType  = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
		buffer_device_address_info.buffer = m_buffer;
		m_handle.deviceAddress            = vkGetBufferDeviceAddress(p_device->GetDevice(), &buffer_device_address_info);
		m_handle.stride                   = handle_size_aligned;
		m_handle.size                     = handle_count * handle_size_aligned;

		m_mapped_data = static_cast<uint8_t *>(allocation_info.pMappedData);
	}

	~ShaderBindingTableInfo()
	{
		if (m_buffer && m_allocation)
		{
			vmaDestroyBuffer(p_device->GetAllocator(), m_buffer, m_allocation);
		}
	}

	uint8_t *GetData()
	{
		return m_mapped_data;
	}

	const VkStridedDeviceAddressRegionKHR *GetHandle() const
	{
		return &m_handle;
	}

  private:
	Device *p_device = nullptr;

	uint32_t m_handle_count = 0;
	uint8_t *m_mapped_data  = nullptr;

	VkStridedDeviceAddressRegionKHR m_handle     = {};
	VkBuffer                        m_buffer     = VK_NULL_HANDLE;
	VmaAllocation                   m_allocation = VK_NULL_HANDLE;
	VkDeviceMemory                  m_memory     = VK_NULL_HANDLE;
};

struct ShaderBindingTableInfos
{
	std::unique_ptr<ShaderBindingTableInfo> raygen   = nullptr;
	std::unique_ptr<ShaderBindingTableInfo> miss     = nullptr;
	std::unique_ptr<ShaderBindingTableInfo> hit      = nullptr;
	std::unique_ptr<ShaderBindingTableInfo> callable = nullptr;
};

static VkPipelineCache                                                          PipelineCache;
static std::unordered_map<size_t, VkPipeline>                                   Pipelines;
static std::unordered_map<size_t, VkPipelineLayout>                             PipelineLayouts;
static std::unordered_map<VkPipeline, std::unique_ptr<ShaderBindingTableInfos>> ShaderBindingTables;

static uint32_t PipelineCount = 0;

PipelineState::PipelineState(RHIDevice *device) :
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
}

PipelineState ::~PipelineState()
{
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

		ShaderBindingTables.clear();
		Pipelines.clear();
		PipelineLayouts.clear();

		if (PipelineCache)
		{
			vkDestroyPipelineCache(static_cast<Device *>(p_device)->GetDevice(), PipelineCache, nullptr);
			PipelineCache = VK_NULL_HANDLE;
		}
	}
}

VkPipelineLayout PipelineState::GetPipelineLayout(Descriptor *descriptor)
{
	size_t hash = 0;
	HashCombine(hash, descriptor->GetShaderMeta().hash, GetHash());

	if (PipelineLayouts.find(hash) != PipelineLayouts.end())
	{
		return PipelineLayouts[hash];
	}

	return CreatePipelineLayout(descriptor);
}

VkPipeline PipelineState::GetPipeline(Descriptor *descriptor, RenderTarget *render_target)
{
	for (const auto &[stage, shader] : m_shaders)
	{
		if (stage & RHIShaderStage::Fragment)
		{
			ASSERT(render_target != nullptr);
			return CreateGraphicsPipeline(descriptor, render_target);
		}
		else if (stage & RHIShaderStage::Compute)
		{
			return CreateComputePipeline(descriptor);
		}
		else if (stage & RHIShaderStage::RayGen)
		{
			return CreateRayTracingPipeline(descriptor);
		}
	}
	return VK_NULL_HANDLE;
}

ShaderBindingTable PipelineState::GetShaderBindingTable(VkPipeline pipeline)
{
	ShaderBindingTable sbt;
	if (ShaderBindingTables.find(pipeline) != ShaderBindingTables.end())
	{
		auto &shader_binding_table_infos = ShaderBindingTables.at(pipeline);

		sbt.raygen   = shader_binding_table_infos->raygen->GetHandle();
		sbt.hit      = shader_binding_table_infos->hit->GetHandle();
		sbt.miss     = shader_binding_table_infos->miss->GetHandle();
		sbt.callable = shader_binding_table_infos->callable->GetHandle();
	}
	return sbt;
}

VkPipelineBindPoint PipelineState::GetPipelineBindPoint() const
{
	for (const auto &[stage, shader] : m_shaders)
	{
		if (stage & RHIShaderStage::Fragment)
		{
			return VK_PIPELINE_BIND_POINT_GRAPHICS;
		}
		else if (stage & RHIShaderStage::Compute)
		{
			return VK_PIPELINE_BIND_POINT_COMPUTE;
		}
		else if (stage & RHIShaderStage::RayGen)
		{
			return VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR;
		}
	}
	return VK_PIPELINE_BIND_POINT_GRAPHICS;
}

VkPipelineLayout PipelineState::CreatePipelineLayout(Descriptor *descriptor)
{
	size_t hash = 0;
	HashCombine(hash, descriptor->GetShaderMeta().hash, GetHash());

	// Push constant merge range
	std::unordered_map<size_t, VkPushConstantRange> push_constant_range_map;
	for (auto &constant : descriptor->GetShaderMeta().constants)
	{
		size_t hash = Hash(constant.size, constant.offset);
		if (push_constant_range_map.find(hash) != push_constant_range_map.end())
		{
			push_constant_range_map[hash].stageFlags |= ToVulkanShaderStages(constant.stage);
		}
		else
		{
			VkPushConstantRange push_constant_range = {};
			push_constant_range.stageFlags          = ToVulkanShaderStages(constant.stage);
			push_constant_range.size                = constant.size;
			push_constant_range.offset              = constant.offset;
			push_constant_range_map.emplace(hash, push_constant_range);
		}
	}

	// Push constant merge stage
	std::unordered_map<VkShaderStageFlags, VkPushConstantRange> push_constant_map;
	for (auto &[hash, constant] : push_constant_range_map)
	{
		if (push_constant_map.find(constant.stageFlags) == push_constant_map.end())
		{
			push_constant_map[constant.stageFlags] = constant;
		}
		else
		{
			push_constant_map[constant.stageFlags].offset = std::min(push_constant_map[constant.stageFlags].offset, constant.offset);
			push_constant_map[constant.stageFlags].size += constant.size;
		}
	}

	std::vector<VkPushConstantRange> push_constants;
	push_constants.reserve(push_constant_map.size());
	for (auto &[hash, constant] : push_constant_map)
	{
		push_constants.push_back(std::move(constant));
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

	VkPipelineLayout layout = VK_NULL_HANDLE;
	vkCreatePipelineLayout(static_cast<Device *>(p_device)->GetDevice(), &pipeline_layout_create_info, nullptr, &layout);

	PipelineLayouts.emplace(hash, layout);

	return layout;
}

VkPipeline PipelineState::CreateGraphicsPipeline(Descriptor *descriptor, RenderTarget *render_target)
{
	size_t hash = 0;
	HashCombine(hash, descriptor->GetShaderMeta().hash, GetHash());

	bool dynamic_rendering = static_cast<Device *>(p_device)->IsFeatureSupport(VulkanFeature::DynamicRendering);

	if (render_target)
	{
		if (dynamic_rendering)
		{
			HashCombine(hash, render_target->GetFormatHash());
		}
		else
		{
			HashCombine(hash, render_target->GetRenderPass());
		}
	}

	if (Pipelines.find(hash) != Pipelines.end())
	{
		return Pipelines[hash];
	}

	// Input Assembly State
	VkPipelineInputAssemblyStateCreateInfo input_assembly_state_create_info = {};
	input_assembly_state_create_info.sType                                  = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
	input_assembly_state_create_info.topology                               = ToVulkanPrimitiveTopology[m_input_assembly_state.topology];
	input_assembly_state_create_info.flags                                  = 0;
	input_assembly_state_create_info.primitiveRestartEnable                 = VK_FALSE;

	// Rasterization State
	VkPipelineRasterizationStateCreateInfo rasterization_state_create_info = {};
	rasterization_state_create_info.sType                                  = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
	rasterization_state_create_info.polygonMode                            = ToVulkanPolygonMode[m_rasterization_state.polygon_mode];
	rasterization_state_create_info.cullMode                               = ToVulkanCullMode[m_rasterization_state.cull_mode];
	rasterization_state_create_info.frontFace                              = ToVulkanFrontFace[m_rasterization_state.front_face];
	rasterization_state_create_info.flags                                  = 0;
	rasterization_state_create_info.depthBiasEnable                        = VK_TRUE;
	rasterization_state_create_info.lineWidth                              = 1;

	// Color Blend State
	std::vector<VkPipelineColorBlendAttachmentState> color_blend_attachment_states(m_blend_state.attachment_states.size());

	for (uint32_t i = 0; i < color_blend_attachment_states.size(); i++)
	{
		color_blend_attachment_states[i].blendEnable         = m_blend_state.attachment_states[i].blend_enable;
		color_blend_attachment_states[i].srcColorBlendFactor = ToVulkanBlendFactor[m_blend_state.attachment_states[i].src_color_blend];
		color_blend_attachment_states[i].dstColorBlendFactor = ToVulkanBlendFactor[m_blend_state.attachment_states[i].dst_color_blend];
		color_blend_attachment_states[i].colorBlendOp        = ToVulkanBlendOp[m_blend_state.attachment_states[i].color_blend_op];
		color_blend_attachment_states[i].srcAlphaBlendFactor = ToVulkanBlendFactor[m_blend_state.attachment_states[i].src_alpha_blend];
		color_blend_attachment_states[i].dstAlphaBlendFactor = ToVulkanBlendFactor[m_blend_state.attachment_states[i].dst_alpha_blend];
		color_blend_attachment_states[i].alphaBlendOp        = ToVulkanBlendOp[m_blend_state.attachment_states[i].alpha_blend_op];
		color_blend_attachment_states[i].colorWriteMask      = m_blend_state.attachment_states[i].color_write_mask;
	}

	VkPipelineColorBlendStateCreateInfo color_blend_state_create_info = {};
	color_blend_state_create_info.sType                               = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
	color_blend_state_create_info.logicOpEnable                       = m_blend_state.enable;
	color_blend_state_create_info.logicOp                             = ToVulkanLogicOp[m_blend_state.logic_op];
	color_blend_state_create_info.attachmentCount                     = static_cast<uint32_t>(color_blend_attachment_states.size());
	color_blend_state_create_info.pAttachments                        = color_blend_attachment_states.data();
	color_blend_state_create_info.blendConstants[0]                   = m_blend_state.blend_constants[0];
	color_blend_state_create_info.blendConstants[1]                   = m_blend_state.blend_constants[1];
	color_blend_state_create_info.blendConstants[2]                   = m_blend_state.blend_constants[2];
	color_blend_state_create_info.blendConstants[3]                   = m_blend_state.blend_constants[3];

	// Depth Stencil State
	VkPipelineDepthStencilStateCreateInfo depth_stencil_state_create_info = {};
	depth_stencil_state_create_info.sType                                 = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
	depth_stencil_state_create_info.depthTestEnable                       = m_depth_stencil_state.depth_test_enable;
	depth_stencil_state_create_info.depthWriteEnable                      = m_depth_stencil_state.depth_write_enable;
	depth_stencil_state_create_info.depthCompareOp                        = ToVulkanCompareOp[m_depth_stencil_state.compare];
	// TODO: stencil test
	depth_stencil_state_create_info.back              = VkStencilOpState{};
	depth_stencil_state_create_info.front             = VkStencilOpState{};
	depth_stencil_state_create_info.stencilTestEnable = VK_FALSE;

	// Viewport State
	VkPipelineViewportStateCreateInfo viewport_state_create_info = {};
	viewport_state_create_info.sType                             = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
	viewport_state_create_info.viewportCount                     = 1;
	viewport_state_create_info.scissorCount                      = 1;
	viewport_state_create_info.flags                             = 0;

	// Multisample State
	VkPipelineMultisampleStateCreateInfo multisample_state_create_info = {};
	multisample_state_create_info.sType                                = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
	multisample_state_create_info.rasterizationSamples                 = ToVulkanSampleCount[m_multisample_state.samples];
	multisample_state_create_info.sampleShadingEnable                  = m_multisample_state.enable;
	multisample_state_create_info.pSampleMask                          = m_multisample_state.enable ? &m_multisample_state.sample_mask : nullptr;
	multisample_state_create_info.flags                                = 0;

	// Dynamic State
	std::vector<VkDynamicState>      dynamic_states            = {VK_DYNAMIC_STATE_SCISSOR, VK_DYNAMIC_STATE_VIEWPORT};
	VkPipelineDynamicStateCreateInfo dynamic_state_create_info = {};
	dynamic_state_create_info.sType                            = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
	dynamic_state_create_info.pDynamicStates                   = dynamic_states.data();
	dynamic_state_create_info.dynamicStateCount                = static_cast<uint32_t>(dynamic_states.size());
	dynamic_state_create_info.flags                            = 0;

	// Vertex Input State
	std::vector<VkVertexInputAttributeDescription> attribute_descriptions = {};
	std::vector<VkVertexInputBindingDescription>   binding_descriptions   = {};

	for (auto &attribute : m_vertex_input_state.input_attributes)
	{
		attribute_descriptions.push_back(VkVertexInputAttributeDescription{
		    attribute.location,
		    attribute.binding,
		    ToVulkanFormat[attribute.format],
		    attribute.offset});
	}

	for (auto &binding : m_vertex_input_state.input_bindings)
	{
		binding_descriptions.push_back(VkVertexInputBindingDescription{
		    binding.binding,
		    binding.stride,
		    ToVulkanVertexInputRate[binding.rate]});
	}

	VkPipelineVertexInputStateCreateInfo vertex_input_state_create_info = {};
	vertex_input_state_create_info.sType                                = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
	vertex_input_state_create_info.vertexAttributeDescriptionCount      = static_cast<uint32_t>(attribute_descriptions.size());
	vertex_input_state_create_info.pVertexAttributeDescriptions         = attribute_descriptions.data();
	vertex_input_state_create_info.vertexBindingDescriptionCount        = static_cast<uint32_t>(binding_descriptions.size());
	vertex_input_state_create_info.pVertexBindingDescriptions           = binding_descriptions.data();

	// Shader Stage State
	std::vector<VkPipelineShaderStageCreateInfo> pipeline_shader_stage_create_infos;
	for (auto &[stage, shader] : m_shaders)
	{
		VkPipelineShaderStageCreateInfo shader_stage_create_info = {};

		shader_stage_create_info.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		shader_stage_create_info.stage  = ToVulkanShaderStage[stage];
		shader_stage_create_info.module = static_cast<Shader *>(shader)->GetHandle();
		shader_stage_create_info.pName  = shader->GetEntryPoint().c_str();
		pipeline_shader_stage_create_infos.push_back(shader_stage_create_info);
	}

	VkGraphicsPipelineCreateInfo graphics_pipeline_create_info = {};
	graphics_pipeline_create_info.sType                        = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
	graphics_pipeline_create_info.stageCount                   = static_cast<uint32_t>(pipeline_shader_stage_create_infos.size());
	graphics_pipeline_create_info.pStages                      = pipeline_shader_stage_create_infos.data();

	graphics_pipeline_create_info.pInputAssemblyState = &input_assembly_state_create_info;
	graphics_pipeline_create_info.pRasterizationState = &rasterization_state_create_info;
	graphics_pipeline_create_info.pColorBlendState    = &color_blend_state_create_info;
	graphics_pipeline_create_info.pViewportState      = &viewport_state_create_info;
	graphics_pipeline_create_info.pMultisampleState   = &multisample_state_create_info;
	graphics_pipeline_create_info.pDynamicState       = &dynamic_state_create_info;
	graphics_pipeline_create_info.pVertexInputState   = &vertex_input_state_create_info;
	graphics_pipeline_create_info.pDepthStencilState  = &depth_stencil_state_create_info;

	graphics_pipeline_create_info.layout             = GetPipelineLayout(descriptor);
	graphics_pipeline_create_info.renderPass         = VK_NULL_HANDLE;
	graphics_pipeline_create_info.subpass            = 0;
	graphics_pipeline_create_info.basePipelineHandle = VK_NULL_HANDLE;
	graphics_pipeline_create_info.basePipelineIndex  = -1;

	if (dynamic_rendering)
	{
		VkPipelineRenderingCreateInfo pipeline_rendering_create_info = {};
		pipeline_rendering_create_info.sType                         = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
		pipeline_rendering_create_info.colorAttachmentCount          = static_cast<uint32_t>(render_target->GetColorFormats().size());
		pipeline_rendering_create_info.pColorAttachmentFormats       = render_target->GetColorFormats().data();
		pipeline_rendering_create_info.depthAttachmentFormat         = render_target->GetDepthFormat().has_value() ? render_target->GetDepthFormat().value() : VK_FORMAT_UNDEFINED;
		pipeline_rendering_create_info.stencilAttachmentFormat       = render_target->GetStencilFormat().has_value() ? render_target->GetStencilFormat().value() : VK_FORMAT_UNDEFINED;
		graphics_pipeline_create_info.pNext                          = &pipeline_rendering_create_info;
	}
	else
	{
		if (render_target)
		{
			graphics_pipeline_create_info.renderPass = render_target->GetRenderPass();
		}
	}

	VkPipeline pipeline = VK_NULL_HANDLE;
	vkCreateGraphicsPipelines(static_cast<Device *>(p_device)->GetDevice(), PipelineCache, 1, &graphics_pipeline_create_info, nullptr, &pipeline);

	Pipelines.emplace(hash, pipeline);

	return pipeline;
}

VkPipeline PipelineState::CreateComputePipeline(Descriptor *descriptor)
{
	size_t hash = 0;
	HashCombine(hash, descriptor->GetShaderMeta().hash, GetHash());

	if (Pipelines.find(hash) != Pipelines.end())
	{
		return Pipelines[hash];
	}

	VkPipelineShaderStageCreateInfo shader_stage_create_info = {};
	for (const auto &[stage, shader] : m_shaders)
	{
		if (stage & RHIShaderStage::Compute)
		{
			shader_stage_create_info.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
			shader_stage_create_info.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
			shader_stage_create_info.module = static_cast<const Shader *>(shader)->GetHandle();
			shader_stage_create_info.pName  = shader->GetEntryPoint().c_str();
			break;
		}
	}

	VkComputePipelineCreateInfo compute_pipeline_create_info = {};
	compute_pipeline_create_info.sType                       = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
	compute_pipeline_create_info.stage                       = shader_stage_create_info;
	compute_pipeline_create_info.layout                      = GetPipelineLayout(descriptor);
	compute_pipeline_create_info.basePipelineIndex           = 0;
	compute_pipeline_create_info.basePipelineHandle          = VK_NULL_HANDLE;

	VkPipeline pipeline = VK_NULL_HANDLE;
	vkCreateComputePipelines(static_cast<Device *>(p_device)->GetDevice(), PipelineCache, 1, &compute_pipeline_create_info, nullptr, &pipeline);

	Pipelines.emplace(hash, pipeline);

	return pipeline;
}

VkPipeline PipelineState::CreateRayTracingPipeline(Descriptor *descriptor)
{
	size_t hash = 0;
	HashCombine(hash, descriptor->GetShaderMeta().hash, GetHash());

	if (Pipelines.find(hash) != Pipelines.end())
	{
		return Pipelines[hash];
	}

	VkPipeline pipeline = VK_NULL_HANDLE;

	std::vector<VkRayTracingShaderGroupCreateInfoKHR> shader_group_create_infos;
	std::vector<VkPipelineShaderStageCreateInfo>      pipeline_shader_stage_create_infos;

	uint32_t raygen_count   = 0;
	uint32_t raymiss_count  = 0;
	uint32_t rayhit_count   = 0;
	uint32_t callable_count = 0;

	// Ray Generation Group
	{
		for (const auto &[stage, shader] : m_shaders)
		{
			if (stage & RHIShaderStage::RayGen)
			{
				VkPipelineShaderStageCreateInfo pipeline_shader_stage_create_info = {};
				pipeline_shader_stage_create_info.sType                           = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
				pipeline_shader_stage_create_info.stage                           = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
				pipeline_shader_stage_create_info.module                          = static_cast<const Shader *>(shader)->GetHandle();
				pipeline_shader_stage_create_info.pName                           = shader->GetEntryPoint().c_str();
				pipeline_shader_stage_create_infos.push_back(pipeline_shader_stage_create_info);

				VkRayTracingShaderGroupCreateInfoKHR shader_group = {};
				shader_group.sType                                = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
				shader_group.type                                 = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
				shader_group.generalShader                        = static_cast<uint32_t>(pipeline_shader_stage_create_infos.size()) - 1;
				shader_group.closestHitShader                     = VK_SHADER_UNUSED_KHR;
				shader_group.anyHitShader                         = VK_SHADER_UNUSED_KHR;
				shader_group.intersectionShader                   = VK_SHADER_UNUSED_KHR;
				shader_group_create_infos.push_back(shader_group);

				raygen_count++;
			}
		}
	}

	// Ray Miss Group
	{
		for (const auto &[stage, shader] : m_shaders)
		{
			if (stage & RHIShaderStage::Miss)
			{
				VkPipelineShaderStageCreateInfo pipeline_shader_stage_create_info = {};
				pipeline_shader_stage_create_info.sType                           = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
				pipeline_shader_stage_create_info.stage                           = VK_SHADER_STAGE_MISS_BIT_KHR;
				pipeline_shader_stage_create_info.module                          = static_cast<const Shader *>(shader)->GetHandle();
				pipeline_shader_stage_create_info.pName                           = shader->GetEntryPoint().c_str();
				pipeline_shader_stage_create_infos.push_back(pipeline_shader_stage_create_info);

				VkRayTracingShaderGroupCreateInfoKHR shader_group = {};
				shader_group.sType                                = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
				shader_group.type                                 = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
				shader_group.generalShader                        = static_cast<uint32_t>(pipeline_shader_stage_create_infos.size()) - 1;
				shader_group.closestHitShader                     = VK_SHADER_UNUSED_KHR;
				shader_group.anyHitShader                         = VK_SHADER_UNUSED_KHR;
				shader_group.intersectionShader                   = VK_SHADER_UNUSED_KHR;
				shader_group_create_infos.push_back(shader_group);

				raymiss_count++;
			}
		}
	}

	// Closest Hit Group
	{
		for (const auto &[stage, shader] : m_shaders)
		{
			if (stage & RHIShaderStage::ClosestHit)
			{
				VkPipelineShaderStageCreateInfo pipeline_shader_stage_create_info = {};
				pipeline_shader_stage_create_info.sType                           = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
				pipeline_shader_stage_create_info.stage                           = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
				pipeline_shader_stage_create_info.module                          = static_cast<const Shader *>(shader)->GetHandle();
				pipeline_shader_stage_create_info.pName                           = shader->GetEntryPoint().c_str();
				pipeline_shader_stage_create_infos.push_back(pipeline_shader_stage_create_info);

				VkRayTracingShaderGroupCreateInfoKHR shader_group = {};
				shader_group.sType                                = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
				shader_group.type                                 = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
				shader_group.generalShader                        = VK_SHADER_UNUSED_KHR;
				shader_group.closestHitShader                     = static_cast<uint32_t>(pipeline_shader_stage_create_infos.size()) - 1;
				shader_group.anyHitShader                         = VK_SHADER_UNUSED_KHR;
				shader_group.intersectionShader                   = VK_SHADER_UNUSED_KHR;
				shader_group_create_infos.push_back(shader_group);

				rayhit_count++;
			}
		}
	}

	// Any Hit Group
	{
		for (const auto &[stage, shader] : m_shaders)
		{
			if (stage & RHIShaderStage::AnyHit)
			{
				VkPipelineShaderStageCreateInfo pipeline_shader_stage_create_info = {};
				pipeline_shader_stage_create_info.sType                           = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
				pipeline_shader_stage_create_info.stage                           = VK_SHADER_STAGE_ANY_HIT_BIT_KHR;
				pipeline_shader_stage_create_info.module                          = static_cast<const Shader *>(shader)->GetHandle();
				pipeline_shader_stage_create_info.pName                           = shader->GetEntryPoint().c_str();
				pipeline_shader_stage_create_infos.push_back(pipeline_shader_stage_create_info);

				VkRayTracingShaderGroupCreateInfoKHR shader_group = {};
				shader_group.sType                                = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
				shader_group.type                                 = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
				shader_group.generalShader                        = VK_SHADER_UNUSED_KHR;
				shader_group.closestHitShader                     = VK_SHADER_UNUSED_KHR;
				shader_group.anyHitShader                         = static_cast<uint32_t>(pipeline_shader_stage_create_infos.size()) - 1;
				shader_group.intersectionShader                   = VK_SHADER_UNUSED_KHR;
				shader_group_create_infos.push_back(shader_group);

				rayhit_count++;
			}
		}
	}

	// Intersection Group
	{
		for (const auto &[stage, shader] : m_shaders)
		{
			if (stage & RHIShaderStage::Intersection)
			{
				VkPipelineShaderStageCreateInfo pipeline_shader_stage_create_info = {};
				pipeline_shader_stage_create_info.sType                           = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
				pipeline_shader_stage_create_info.stage                           = VK_SHADER_STAGE_INTERSECTION_BIT_KHR;
				pipeline_shader_stage_create_info.module                          = static_cast<const Shader *>(shader)->GetHandle();
				pipeline_shader_stage_create_info.pName                           = shader->GetEntryPoint().c_str();
				pipeline_shader_stage_create_infos.push_back(pipeline_shader_stage_create_info);

				VkRayTracingShaderGroupCreateInfoKHR shader_group = {};
				shader_group.sType                                = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
				shader_group.type                                 = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
				shader_group.generalShader                        = VK_SHADER_UNUSED_KHR;
				shader_group.closestHitShader                     = VK_SHADER_UNUSED_KHR;
				shader_group.anyHitShader                         = VK_SHADER_UNUSED_KHR;
				shader_group.intersectionShader                   = static_cast<uint32_t>(pipeline_shader_stage_create_infos.size()) - 1;
				shader_group_create_infos.push_back(shader_group);

				rayhit_count++;
			}
		}
	}

	// Callable Group
	{
		for (const auto &[stage, shader] : m_shaders)
		{
			if (stage & RHIShaderStage::Callable)
			{
				VkPipelineShaderStageCreateInfo pipeline_shader_stage_create_info = {};
				pipeline_shader_stage_create_info.sType                           = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
				pipeline_shader_stage_create_info.stage                           = VK_SHADER_STAGE_CALLABLE_BIT_KHR;
				pipeline_shader_stage_create_info.module                          = static_cast<const Shader *>(shader)->GetHandle();
				pipeline_shader_stage_create_info.pName                           = shader->GetEntryPoint().c_str();
				pipeline_shader_stage_create_infos.push_back(pipeline_shader_stage_create_info);

				VkRayTracingShaderGroupCreateInfoKHR shader_group = {};
				shader_group.sType                                = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
				shader_group.type                                 = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
				shader_group.generalShader                        = static_cast<uint32_t>(pipeline_shader_stage_create_infos.size()) - 1;
				shader_group.closestHitShader                     = VK_SHADER_UNUSED_KHR;
				shader_group.anyHitShader                         = VK_SHADER_UNUSED_KHR;
				shader_group.intersectionShader                   = VK_SHADER_UNUSED_KHR;
				shader_group_create_infos.push_back(shader_group);

				callable_count++;
			}
		}
	}

	VkRayTracingPipelineCreateInfoKHR raytracing_pipeline_create_info = {};
	raytracing_pipeline_create_info.sType                             = VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR;
	raytracing_pipeline_create_info.stageCount                        = static_cast<uint32_t>(pipeline_shader_stage_create_infos.size());
	raytracing_pipeline_create_info.pStages                           = pipeline_shader_stage_create_infos.data();
	raytracing_pipeline_create_info.groupCount                        = static_cast<uint32_t>(shader_group_create_infos.size());
	raytracing_pipeline_create_info.pGroups                           = shader_group_create_infos.data();
	raytracing_pipeline_create_info.maxPipelineRayRecursionDepth      = 4;
	raytracing_pipeline_create_info.layout                            = GetPipelineLayout(descriptor);

	vkCreateRayTracingPipelinesKHR(static_cast<Device *>(p_device)->GetDevice(), VK_NULL_HANDLE, PipelineCache, 1, &raytracing_pipeline_create_info, nullptr, &pipeline);

	Pipelines.emplace(hash, pipeline);

	// Create shader binding table
	/*
	    SBT Layout:

	        /-----------\
	        | raygen    |
	        |-----------|
	        | miss        |
	        |-----------|
	        | hit           |
	        |-----------|
	        | callable   |
	        \-----------/

	*/

	auto sbt = std::make_unique<ShaderBindingTableInfos>();

	VkPhysicalDeviceRayTracingPipelinePropertiesKHR raytracing_properties = {};
	raytracing_properties.sType                                           = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR;
	VkPhysicalDeviceProperties2 deviceProperties2                         = {};
	deviceProperties2.sType                                               = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
	deviceProperties2.pNext                                               = &raytracing_properties;
	vkGetPhysicalDeviceProperties2(static_cast<Device *>(p_device)->GetPhysicalDevice(), &deviceProperties2);

	const uint32_t handle_size         = raytracing_properties.shaderGroupHandleSize;
	const uint32_t handle_size_aligned = (raytracing_properties.shaderGroupHandleSize + raytracing_properties.shaderGroupHandleAlignment - 1) & ~(raytracing_properties.shaderGroupHandleAlignment - 1);
	const uint32_t group_count         = static_cast<uint32_t>(shader_group_create_infos.size());
	const uint32_t sbt_size            = group_count * handle_size_aligned;

	std::vector<uint8_t> shader_handle_storage(sbt_size);

	vkGetRayTracingShaderGroupHandlesKHR(static_cast<Device *>(p_device)->GetDevice(), pipeline, 0, group_count, sbt_size, shader_handle_storage.data());

	uint32_t handle_offset = 0;

	// Gen Group
	{
		sbt->raygen = std::make_unique<ShaderBindingTableInfo>(static_cast<Device *>(p_device), raygen_count);
		std::memcpy(sbt->raygen->GetData(), shader_handle_storage.data() + handle_offset, handle_size * raygen_count);
		handle_offset += raygen_count * handle_size_aligned;
	}

	// Miss Group
	{
		sbt->miss = std::make_unique<ShaderBindingTableInfo>(static_cast<Device *>(p_device), raymiss_count);
		std::memcpy(sbt->miss->GetData(), shader_handle_storage.data() + handle_offset, handle_size * raymiss_count);
		handle_offset += raymiss_count * handle_size_aligned;
	}

	// Hit Group
	{
		sbt->hit = std::make_unique<ShaderBindingTableInfo>(static_cast<Device *>(p_device), rayhit_count);
		std::memcpy(sbt->hit->GetData(), shader_handle_storage.data() + handle_offset, handle_size * rayhit_count);
		handle_offset += rayhit_count * handle_size_aligned;
	}

	// Callable Group
	{
		sbt->callable = std::make_unique<ShaderBindingTableInfo>(static_cast<Device *>(p_device), callable_count);
		std::memcpy(sbt->callable->GetData(), shader_handle_storage.data() + handle_offset, handle_size * callable_count);
		handle_offset += callable_count * handle_size_aligned;
	}

	ShaderBindingTables.emplace(pipeline, std::move(sbt));

	return pipeline;
}
}        // namespace Ilum::Vulkan