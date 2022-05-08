#define GLFW_INCLUDE_VULKAN
#include <Core/Window.hpp>

#define VMA_IMPLEMENTATION
#define VOLK_IMPLEMENTATION
#include "AccelerateStructure.hpp"
#include "Buffer.hpp"
#include "Command.hpp"
#include "DescriptorAllocator.hpp"
#include "DescriptorState.hpp"
#include "Device.hpp"
#include "Frame.hpp"
#include "FrameBuffer.hpp"
#include "PipelineState.hpp"
#include "ShaderAllocator.hpp"
#include "ShaderBindingTable.hpp"
#include "Texture.hpp"

#include <map>
#include <optional>
#include <sstream>

namespace Ilum
{
PFN_vkCreateDebugUtilsMessengerEXT          RHIDevice::vkCreateDebugUtilsMessengerEXT          = nullptr;
VkDebugUtilsMessengerEXT                    RHIDevice::vkDebugUtilsMessengerEXT                = nullptr;
PFN_vkDestroyDebugUtilsMessengerEXT         RHIDevice::vkDestroyDebugUtilsMessengerEXT         = nullptr;
PFN_vkSetDebugUtilsObjectTagEXT             RHIDevice::vkSetDebugUtilsObjectTagEXT             = nullptr;
PFN_vkSetDebugUtilsObjectNameEXT            RHIDevice::vkSetDebugUtilsObjectNameEXT            = nullptr;
PFN_vkCmdBeginDebugUtilsLabelEXT            RHIDevice::vkCmdBeginDebugUtilsLabelEXT            = nullptr;
PFN_vkCmdEndDebugUtilsLabelEXT              RHIDevice::vkCmdEndDebugUtilsLabelEXT              = nullptr;
PFN_vkGetPhysicalDeviceMemoryProperties2KHR RHIDevice::vkGetPhysicalDeviceMemoryProperties2KHR = nullptr;

#ifdef _DEBUG
const std::vector<const char *>                 RHIDevice::s_instance_extensions   = {"VK_KHR_surface", "VK_KHR_win32_surface", "VK_EXT_debug_report", "VK_EXT_debug_utils"};
const std::vector<const char *>                 RHIDevice::s_validation_layers     = {"VK_LAYER_KHRONOS_validation"};
const std::vector<VkValidationFeatureEnableEXT> RHIDevice::s_validation_extensions = {
    VK_VALIDATION_FEATURE_ENABLE_GPU_ASSISTED_RESERVE_BINDING_SLOT_EXT,
    VK_VALIDATION_FEATURE_ENABLE_GPU_ASSISTED_EXT,
    VK_VALIDATION_FEATURE_ENABLE_SYNCHRONIZATION_VALIDATION_EXT};
#else
const std::vector<const char *>                 RHIDevice::s_instance_extensions   = {"VK_KHR_surface", "VK_KHR_win32_surface"};
const std::vector<const char *>                 RHIDevice::s_validation_layers     = {};
const std::vector<VkValidationFeatureEnableEXT> RHIDevice::s_validation_extensions = {};
#endif        // _DEBUG

const std::vector<const char *> RHIDevice::s_device_extensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME,
    VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
    VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME,
    VK_KHR_RAY_QUERY_EXTENSION_NAME,
    VK_NV_MESH_SHADER_EXTENSION_NAME,
    VK_EXT_SHADER_VIEWPORT_INDEX_LAYER_EXTENSION_NAME,
    VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
    VK_KHR_SHADER_DRAW_PARAMETERS_EXTENSION_NAME};

inline const std::vector<const char *> GetInstanceExtensionSupported(const std::vector<const char *> &extensions)
{
	uint32_t extension_count = 0;
	vkEnumerateInstanceExtensionProperties(nullptr, &extension_count, nullptr);

	std::vector<VkExtensionProperties> device_extensions(extension_count);
	vkEnumerateInstanceExtensionProperties(nullptr, &extension_count, device_extensions.data());

	std::vector<const char *> result;

	for (const auto &extension : extensions)
	{
		bool found = false;
		for (const auto &device_extension : device_extensions)
		{
			if (strcmp(extension, device_extension.extensionName) == 0)
			{
				result.emplace_back(extension);
				found = true;
				LOG_INFO("Enable instance extension: {}", extension);
				break;
			}
		}
		if (!found)
		{
			LOG_WARN("Instance extension {} is not supported", extension);
		}
	}

	return result;
}

inline bool CheckLayerSupported(const char *layer_name)
{
	uint32_t layer_count;
	vkEnumerateInstanceLayerProperties(&layer_count, nullptr);

	std::vector<VkLayerProperties> layers(layer_count);
	vkEnumerateInstanceLayerProperties(&layer_count, layers.data());

	for (const auto &layer : layers)
	{
		if (strcmp(layer.layerName, layer_name) == 0)
		{
			return true;
		}
	}

	return false;
}

static inline VKAPI_ATTR VkBool32 VKAPI_CALL ValidationCallback(VkDebugUtilsMessageSeverityFlagBitsEXT msg_severity, VkDebugUtilsMessageTypeFlagsEXT msg_type, const VkDebugUtilsMessengerCallbackDataEXT *callback_data, void *user_data)
{
	if (msg_severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT)
	{
		LOG_INFO(callback_data->pMessage);
	}
	else if (msg_severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT)
	{
		LOG_WARN(callback_data->pMessage);
	}
	else if (msg_severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT)
	{
		LOG_ERROR(callback_data->pMessage);
	}

	return VK_FALSE;
}

inline uint32_t ScorePhysicalDevice(VkPhysicalDevice physical_device, const std::vector<const char *> &device_extensions)
{
	uint32_t score = 0;

	// Check extensions
	uint32_t device_extension_properties_count = 0;
	vkEnumerateDeviceExtensionProperties(physical_device, nullptr, &device_extension_properties_count, nullptr);

	std::vector<VkExtensionProperties> extension_properties(device_extension_properties_count);
	vkEnumerateDeviceExtensionProperties(physical_device, nullptr, &device_extension_properties_count, extension_properties.data());

	for (auto &device_extension : device_extensions)
	{
		for (auto &support_extension : extension_properties)
		{
			if (std::strcmp(device_extension, support_extension.extensionName) == 0)
			{
				score += 100;
				break;
			}
		}
	}

	VkPhysicalDeviceProperties       properties        = {};
	VkPhysicalDeviceFeatures         features          = {};
	VkPhysicalDeviceMemoryProperties memory_properties = {};

	vkGetPhysicalDeviceProperties(physical_device, &properties);
	vkGetPhysicalDeviceFeatures(physical_device, &features);
	vkGetPhysicalDeviceMemoryProperties(physical_device, &memory_properties);

	// Logging gpu
	std::stringstream ss;
	ss << "\nFound physical device [" << properties.deviceID << "]\n";
	ss << "Name: " << properties.deviceName << "\n";
	ss << "Type: ";
	switch (static_cast<int32_t>(properties.deviceType))
	{
		case 1:
			ss << "Integrated\n";
			break;
		case 2:
			ss << "Discrete\n";
			break;
		case 3:
			ss << "Virtual\n";
			break;
		case 4:
			ss << "CPU\n";
			break;
		default:
			ss << "Other " << properties.deviceType << "\n";
	}

	ss << "Vendor: ";
	switch (properties.vendorID)
	{
		case 0x8086:
			ss << "Intel\n";
			break;
		case 0x10DE:
			ss << "Nvidia\n";
			break;
		case 0x1002:
			ss << "AMD\n";
			break;
		default:
			ss << properties.vendorID << "\n";
	}

	uint32_t supportedVersion[3] = {
	    VK_VERSION_MAJOR(properties.apiVersion),
	    VK_VERSION_MINOR(properties.apiVersion),
	    VK_VERSION_PATCH(properties.apiVersion)};

	ss << "API Version: " << supportedVersion[0] << "." << supportedVersion[1] << "." << supportedVersion[2] << '\n';

	// Score discrete gpu
	if (properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)
	{
		score += 1000;
	}

	score += properties.limits.maxImageDimension2D;
	return score;
}

inline VkPhysicalDevice SelectPhysicalDevice(const std::vector<VkPhysicalDevice> &physical_devices, const std::vector<const char *> &device_extensions)
{
	// Score - GPU
	std::map<uint32_t, VkPhysicalDevice, std::greater<uint32_t>> scores;
	for (auto &gpu : physical_devices)
	{
		scores.emplace(ScorePhysicalDevice(gpu, device_extensions), gpu);
	}

	if (scores.empty())
	{
		return VK_NULL_HANDLE;
	}

	return scores.begin()->second;
}

inline const std::vector<const char *> GetDeviceExtensionSupport(VkPhysicalDevice physical_device, const std::vector<const char *> &extensions)
{
	std::vector<const char *> result;

	uint32_t device_extension_properties_count = 0;
	vkEnumerateDeviceExtensionProperties(physical_device, nullptr, &device_extension_properties_count, nullptr);

	std::vector<VkExtensionProperties> extension_properties(device_extension_properties_count);
	vkEnumerateDeviceExtensionProperties(physical_device, nullptr, &device_extension_properties_count, extension_properties.data());

	for (auto &device_extension : extensions)
	{
		bool enable = false;
		for (auto &support_extension : extension_properties)
		{
			if (std::strcmp(device_extension, support_extension.extensionName) == 0)
			{
				result.push_back(device_extension);
				enable = true;
				LOG_INFO("Enable device extension: {}", device_extension);
				break;
			}
		}

		if (!enable)
		{
			LOG_WARN("Device extension {} is not supported", device_extension);
		}
	}

	return result;
}

inline std::optional<uint32_t> GetQueueFamilyIndex(const std::vector<VkQueueFamilyProperties> &queue_family_properties, VkQueueFlagBits queue_flag)
{
	// Dedicated queue for compute
	// Try to find a queue family index that supports compute but not graphics
	if (queue_flag & VK_QUEUE_COMPUTE_BIT)
	{
		for (uint32_t i = 0; i < static_cast<uint32_t>(queue_family_properties.size()); i++)
		{
			if ((queue_family_properties[i].queueFlags & queue_flag) && ((queue_family_properties[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) == 0))
			{
				return i;
				break;
			}
		}
	}

	// Dedicated queue for transfer
	// Try to find a queue family index that supports transfer but not graphics and compute
	if (queue_flag & VK_QUEUE_TRANSFER_BIT)
	{
		for (uint32_t i = 0; i < static_cast<uint32_t>(queue_family_properties.size()); i++)
		{
			if ((queue_family_properties[i].queueFlags & queue_flag) && ((queue_family_properties[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) == 0) && ((queue_family_properties[i].queueFlags & VK_QUEUE_COMPUTE_BIT) == 0))
			{
				return i;
				break;
			}
		}
	}

	// For other queue types or if no separate compute queue is present, return the first one to support the requested flags
	for (uint32_t i = 0; i < static_cast<uint32_t>(queue_family_properties.size()); i++)
	{
		if (queue_family_properties[i].queueFlags & queue_flag)
		{
			return i;
			break;
		}
	}

	return std::optional<uint32_t>();
}

VkFormat FindSupportedFormat(VkPhysicalDevice physical_device, const std::vector<VkFormat> &candidates, VkImageTiling tiling, VkFormatFeatureFlags features)
{
	for (VkFormat format : candidates)
	{
		VkFormatProperties props;
		vkGetPhysicalDeviceFormatProperties(physical_device, format, &props);

		if (tiling == VK_IMAGE_TILING_LINEAR && (props.linearTilingFeatures & features) == features)
		{
			return format;
		}
		else if (tiling == VK_IMAGE_TILING_OPTIMAL && (props.optimalTilingFeatures & features) == features)
		{
			return format;
		}
	}

	throw std::runtime_error("findSupportedFormat failed");
}

VkFormat FindDepthFormat(VkPhysicalDevice physical_device)
{
	return FindSupportedFormat(physical_device, {VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT},
	                           VK_IMAGE_TILING_OPTIMAL,
	                           VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT);
}

RHIDevice::RHIDevice(Window *window) :
    p_window(window)
{
	CreateInstance();
	CreatePhysicalDevice();
	CreateSurface();
	CreateLogicalDevice();
	CreateSwapchain();

	p_window->OnWindowSizeFunc += [this](int32_t width, int32_t height) { 
		if (width != 0 && height != 0)
		{
			CreateSwapchain();
		}
	};

	VkPipelineCacheCreateInfo create_info = {};
	create_info.sType                     = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
	vkCreatePipelineCache(m_device, &create_info, nullptr, &m_pipeline_cache);

	CreateAllocator();
}

RHIDevice::~RHIDevice()
{
	vkDeviceWaitIdle(m_device);

	for (auto &[hash, pipeline] : m_pipelines)
	{
		vkDestroyPipeline(m_device, pipeline, nullptr);
	}

	for (auto &[hash, pipeline_layout] : m_pipeline_layouts)
	{
		vkDestroyPipelineLayout(m_device, pipeline_layout, nullptr);
	}

	for (auto &[hash, render_pass] : m_render_passes)
	{
		vkDestroyRenderPass(m_device, render_pass, nullptr);
	}

	for (auto &[hash, frame_buffer] : m_frame_buffers)
	{
		vkDestroyFramebuffer(m_device, frame_buffer, nullptr);
	}

	m_pipelines.clear();
	m_frame_buffers.clear();
	m_render_passes.clear();
	m_pipeline_layouts.clear();
	m_descriptor_sets.clear();
	m_shader_binding_tables.clear();
	m_descriptor_states.clear();

	m_shader_allocator.reset();
	m_descriptor_allocator.reset();

	if (m_pipeline_cache)
	{
		vkDestroyPipelineCache(m_device, m_pipeline_cache, nullptr);
	}

	m_swapchain_images.clear();

	for (size_t i = 0; i < m_frames.size(); i++)
	{
		m_frames[i]->ReleaseAllocatedSemaphore(m_render_complete[i]);
		m_frames[i]->ReleaseAllocatedSemaphore(m_present_complete[i]);
		m_frames[i]->Reset();
	}

	m_frames.clear();

	if (m_swapchain)
	{
		vkDestroySwapchainKHR(m_device, m_swapchain, nullptr);
	}

	if (m_surface)
	{
		vkDestroySurfaceKHR(m_instance, m_surface, nullptr);
	}

	if (m_allocator)
	{
		vmaDestroyAllocator(m_allocator);
	}

	if (m_device)
	{
		vkDestroyDevice(m_device, nullptr);
	}

	if (vkDestroyDebugUtilsMessengerEXT)
	{
		vkDestroyDebugUtilsMessengerEXT(m_instance, vkDebugUtilsMessengerEXT, nullptr);
	}
	vkDestroyInstance(m_instance, nullptr);
}

VkInstance RHIDevice::GetVulkanInstance() const
{
	return m_instance;
}

VkPhysicalDevice RHIDevice::GetPhysicalDevice() const
{
	return m_physical_device;
}

VkDevice RHIDevice::GetDevice() const
{
	return m_device;
}

VmaAllocator RHIDevice::GetAllocator() const
{
	return m_allocator;
}

VkPipelineCache RHIDevice::GetPipelineCache() const
{
	return m_pipeline_cache;
}

std::vector<Texture *> RHIDevice::GetSwapchainImages() const
{
	std::vector<Texture *> swapchain_images(m_swapchain_images.size());
	for (size_t i = 0; i < swapchain_images.size(); i++)
	{
		swapchain_images[i] = m_swapchain_images[i].get();
	}
	return swapchain_images;
}

Texture *RHIDevice::GetBackBuffer() const
{
	return m_swapchain_images[m_current_frame].get();
}

VkQueue RHIDevice::GetQueue(VkQueueFlagBits flag) const
{
	switch (flag)
	{
		case VK_QUEUE_GRAPHICS_BIT:
			return m_graphics_queue;
		case VK_QUEUE_COMPUTE_BIT:
			return m_compute_queue;
		case VK_QUEUE_TRANSFER_BIT:
			return m_transfer_queue;
	}
	return m_graphics_queue;
}

VkFormat RHIDevice::GetSwapchainFormat() const
{
	return m_swapchain_format;
}

VkFormat RHIDevice::GetDepthStencilFormat() const
{
	return m_depth_format;
}

VkShaderModule RHIDevice::LoadShader(const ShaderDesc &desc)
{
	return m_shader_allocator->Load(desc);
}

ShaderReflectionData RHIDevice::ReflectShader(VkShaderModule shader)
{
	return m_shader_allocator->Reflect(shader);
}

CommandBuffer &RHIDevice::RequestCommandBuffer(VkCommandBufferLevel level, VkQueueFlagBits queue)
{
	return m_frames[m_current_frame]->RequestCommandBuffer(level, queue);
}

VkPipelineLayout RHIDevice::AllocatePipelineLayout(const PipelineState &pso)
{
	size_t hash = pso.Hash();
	if (m_pipeline_layouts.find(hash) != m_pipeline_layouts.end())
	{
		return m_pipeline_layouts.at(hash);
	}

	m_pipeline_layouts[hash] = VK_NULL_HANDLE;

	ShaderReflectionData meta = {};
	for (auto &shader : pso.GetShaders())
	{
		meta += ReflectShader(LoadShader(shader));
	}

	std::vector<VkDescriptorSetLayout> descriptor_set_layouts;
	// Create descriptor layout & set
	for (auto &set : meta.sets)
	{
		descriptor_set_layouts.push_back(m_descriptor_allocator->GetDescriptorLayout(meta, set));
		m_descriptor_sets[hash][set] = m_descriptor_allocator->AllocateDescriptorSet(meta, set);
	}

	// Create pipeline layout
	std::vector<VkPushConstantRange> push_constants;
	for (auto &constant : meta.constants)
	{
		if (constant.type == ShaderReflectionData::Constant::Type::Push)
		{
			VkPushConstantRange push_constant_range = {};
			push_constant_range.stageFlags          = constant.stage;
			push_constant_range.size                = constant.size;
			push_constant_range.offset              = constant.offset;
			push_constants.push_back(push_constant_range);
		}
	}

	VkPipelineLayoutCreateInfo pipeline_layout_create_info = {};
	pipeline_layout_create_info.sType                      = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
	pipeline_layout_create_info.pushConstantRangeCount     = static_cast<uint32_t>(push_constants.size());
	pipeline_layout_create_info.pPushConstantRanges        = push_constants.data();
	pipeline_layout_create_info.setLayoutCount             = static_cast<uint32_t>(descriptor_set_layouts.size());
	pipeline_layout_create_info.pSetLayouts                = descriptor_set_layouts.data();

	vkCreatePipelineLayout(m_device, &pipeline_layout_create_info, nullptr, &m_pipeline_layouts[hash]);
	return m_pipeline_layouts[hash];
}

VkPipeline RHIDevice::AllocatePipeline(const PipelineState &pso, VkRenderPass render_pass)
{
	size_t hash = pso.Hash();
	if (m_pipelines.find(hash) != m_pipelines.end())
	{
		return m_pipelines.at(hash);
	}

	switch (pso.GetBindPoint())
	{
		case VK_PIPELINE_BIND_POINT_GRAPHICS:
			return AllocateGraphicsPipeline(pso, render_pass);
		case VK_PIPELINE_BIND_POINT_COMPUTE:
			return AllocateComputePipeline(pso);
		case VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR:
			return AllocateRayTracingPipeline(pso);
		default:
			break;
	}

	return m_pipelines.at(hash);
}

VkRenderPass RHIDevice::AllocateRenderPass(FrameBuffer &framebuffer)
{
	size_t hash = framebuffer.Hash();

	if (m_render_passes.find(hash) != m_render_passes.end())
	{
		return m_render_passes[hash];
	}

	m_render_passes[hash] = VK_NULL_HANDLE;

	VkSubpassDescription subpass_description    = {};
	subpass_description.pipelineBindPoint       = VK_PIPELINE_BIND_POINT_GRAPHICS;
	subpass_description.colorAttachmentCount    = static_cast<uint32_t>(framebuffer.m_attachment_references.size());
	subpass_description.pColorAttachments       = framebuffer.m_attachment_references.data();
	subpass_description.pDepthStencilAttachment = framebuffer.m_depth_stencil_attachment_reference.has_value() ? &framebuffer.m_depth_stencil_attachment_reference.value() : nullptr;

	std::array<VkSubpassDependency, 2> subpass_dependencies;

	subpass_dependencies[0].srcSubpass      = VK_SUBPASS_EXTERNAL;
	subpass_dependencies[0].dstSubpass      = 0;
	subpass_dependencies[0].srcStageMask    = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
	subpass_dependencies[0].dstStageMask    = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
	subpass_dependencies[0].srcAccessMask   = VK_ACCESS_MEMORY_READ_BIT;
	subpass_dependencies[0].dstAccessMask   = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
	subpass_dependencies[0].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

	subpass_dependencies[1].srcSubpass      = 0;
	subpass_dependencies[1].dstSubpass      = VK_SUBPASS_EXTERNAL;
	subpass_dependencies[1].srcStageMask    = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
	subpass_dependencies[1].dstStageMask    = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
	subpass_dependencies[1].srcAccessMask   = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
	subpass_dependencies[1].dstAccessMask   = VK_ACCESS_MEMORY_READ_BIT;
	subpass_dependencies[1].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

	// Create render pass
	VkRenderPassCreateInfo render_pass_create_info = {};
	render_pass_create_info.sType                  = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
	render_pass_create_info.attachmentCount        = static_cast<uint32_t>(framebuffer.m_attachment_descriptions.size());
	render_pass_create_info.pAttachments           = framebuffer.m_attachment_descriptions.data();
	render_pass_create_info.subpassCount           = 1;
	render_pass_create_info.pSubpasses             = &subpass_description;
	render_pass_create_info.dependencyCount        = static_cast<uint32_t>(subpass_dependencies.size());
	render_pass_create_info.pDependencies          = subpass_dependencies.data();

	vkCreateRenderPass(m_device, &render_pass_create_info, nullptr, &m_render_passes[hash]);

	return m_render_passes[hash];
}

VkFramebuffer RHIDevice::AllocateFrameBuffer(FrameBuffer &framebuffer)
{
	size_t hash = framebuffer.Hash();

	if (m_frame_buffers.find(hash) != m_frame_buffers.end())
	{
		return m_frame_buffers[hash];
	}

	m_frame_buffers[hash] = VK_NULL_HANDLE;

	VkFramebufferCreateInfo frame_buffer_create_info = {};
	frame_buffer_create_info.sType                   = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
	frame_buffer_create_info.renderPass              = AllocateRenderPass(framebuffer);
	frame_buffer_create_info.attachmentCount         = static_cast<uint32_t>(framebuffer.m_views.size());
	frame_buffer_create_info.pAttachments            = framebuffer.m_views.data();
	frame_buffer_create_info.width                   = framebuffer.m_width;
	frame_buffer_create_info.height                  = framebuffer.m_height;
	frame_buffer_create_info.layers                  = framebuffer.m_layer;

	vkCreateFramebuffer(m_device, &frame_buffer_create_info, nullptr, &m_frame_buffers[hash]);

	return m_frame_buffers[hash];
}

ShaderBindingTable &RHIDevice::AllocateSBT(const PipelineState &pso)
{
	size_t hash = pso.Hash();
	if (m_shader_binding_tables.find(hash) != m_shader_binding_tables.end())
	{
		return *m_shader_binding_tables.at(hash);
	}

	AllocatePipeline(pso);

	return *m_shader_binding_tables.at(hash);
}

const std::map<uint32_t, VkDescriptorSet> &RHIDevice::AllocateDescriptorSet(const PipelineState &pso)
{
	size_t hash = pso.Hash();
	if (m_descriptor_sets.find(hash) != m_descriptor_sets.end())
	{
		return m_descriptor_sets.at(hash);
	}

	m_descriptor_sets[hash] = {};

	ShaderReflectionData meta = {};
	for (auto &shader : pso.GetShaders())
	{
		meta += ReflectShader(LoadShader(shader));
	}

	std::vector<VkDescriptorSetLayout> descriptor_set_layouts;
	// Create descriptor layout & set
	for (auto &set : meta.sets)
	{
		descriptor_set_layouts.push_back(m_descriptor_allocator->GetDescriptorLayout(meta, set));
		m_descriptor_sets[hash][set] = m_descriptor_allocator->AllocateDescriptorSet(meta, set);
	}

	return m_descriptor_sets.at(hash);
}

DescriptorState &RHIDevice::AllocateDescriptorState(const PipelineState &pso)
{
	size_t hash = pso.Hash();
	if (m_descriptor_states.find(hash) != m_descriptor_states.end())
	{
		return *m_descriptor_states.at(hash);
	}

	m_descriptor_states[hash] = std::make_unique<DescriptorState>(this, &pso);

	return *m_descriptor_states.at(hash);
}

uint32_t RHIDevice::GetGraphicsFamily() const
{
	return m_graphics_family;
}

uint32_t RHIDevice::GetComputeFamily() const
{
	return m_compute_family;
}

uint32_t RHIDevice::GetTransferFamily() const
{
	return m_transfer_family;
}

uint32_t RHIDevice::GetPresentFamily() const
{
	return m_present_family;
}

uint32_t RHIDevice::GetCurrentFrame() const
{
	return m_current_frame;
}

void RHIDevice::NewFrame()
{
	auto result = vkAcquireNextImageKHR(m_device, m_swapchain, std::numeric_limits<uint64_t>::max(), m_present_complete[m_current_frame], VK_NULL_HANDLE, &m_current_frame);

	if (result == VK_ERROR_OUT_OF_DATE_KHR)
	{
		CreateSwapchain();
		return;
	}

	if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR)
	{
		LOG_ERROR("Failed to acquire swapchain image!");
		return;
	}

	m_frames[m_current_frame]->Reset();
	m_cmd_buffer_for_submit.clear();
}

void RHIDevice::Submit(CommandBuffer &cmd_buffer)
{
	m_cmd_buffer_for_submit.push_back(cmd_buffer);
}

void RHIDevice::SubmitIdle(CommandBuffer &cmd_buffer, VkQueueFlagBits queue)
{
	VkSubmitInfo submit_info          = {};
	submit_info.sType                 = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	submit_info.commandBufferCount    = 1;
	VkCommandBuffer cmd_buffer_handle = cmd_buffer;
	submit_info.pCommandBuffers       = &cmd_buffer_handle;

	VkFence fence = VK_NULL_HANDLE;

	VkFenceCreateInfo fence_create_info = {};
	fence_create_info.sType             = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;

	vkCreateFence(m_device, &fence_create_info, nullptr, &fence);
	vkResetFences(m_device, 1, &fence);

	switch (queue)
	{
		case VK_QUEUE_GRAPHICS_BIT:
			vkQueueWaitIdle(m_graphics_queue);
			vkQueueSubmit(m_graphics_queue, 1, &submit_info, fence);
			break;
		case VK_QUEUE_COMPUTE_BIT:
			vkQueueWaitIdle(m_compute_queue);
			vkQueueSubmit(m_compute_queue, 1, &submit_info, fence);
			break;
		case VK_QUEUE_TRANSFER_BIT:
			vkQueueWaitIdle(m_transfer_queue);
			vkQueueSubmit(m_transfer_queue, 1, &submit_info, fence);
			break;
		default:
			break;
	}

	vkWaitForFences(m_device, 1, &fence, VK_TRUE, std::numeric_limits<uint64_t>::max());
	vkDestroyFence(m_device, fence, nullptr);
}

void RHIDevice::EndFrame()
{
	VkSubmitInfo submit_info         = {};
	submit_info.sType                = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	submit_info.commandBufferCount   = static_cast<uint32_t>(m_cmd_buffer_for_submit.size());
	submit_info.pCommandBuffers      = m_cmd_buffer_for_submit.data();
	VkPipelineStageFlags wait_stages = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
	submit_info.pWaitDstStageMask    = &wait_stages;

	submit_info.waitSemaphoreCount = 1;
	submit_info.pWaitSemaphores    = &m_present_complete[m_current_frame];

	submit_info.signalSemaphoreCount = 1;
	submit_info.pSignalSemaphores    = &m_render_complete[m_current_frame];

	vkQueueSubmit(m_graphics_queue, 1, &submit_info, m_frames[m_current_frame]->RequestFence());

	VkPresentInfoKHR present_info   = {};
	present_info.sType              = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
	present_info.waitSemaphoreCount = 1;
	present_info.pWaitSemaphores    = &m_render_complete[m_current_frame];
	present_info.swapchainCount     = 1;
	present_info.pSwapchains        = &m_swapchain;
	present_info.pImageIndices      = &m_current_frame;

	auto result = vkQueuePresentKHR(m_present_queue, &present_info);

	if (result == VK_ERROR_OUT_OF_DATE_KHR)
	{
		CreateSwapchain();
		return;
	}

	if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR)
	{
		LOG_ERROR("Failed to present swapchain image!");
		return;
	}

	m_current_frame = (m_current_frame + 1) % static_cast<uint32_t>(m_swapchain_images.size());
}

void RHIDevice::Reset()
{
	vkDeviceWaitIdle(m_device);

	for (auto &[hash, render_pass] : m_render_passes)
	{
		vkDestroyRenderPass(m_device, render_pass, nullptr);
	}

	for (auto &[hash, frame_buffer] : m_frame_buffers)
	{
		vkDestroyFramebuffer(m_device, frame_buffer, nullptr);
	}

	m_render_passes.clear();
	m_frame_buffers.clear();
}

void RHIDevice::CreateInstance()
{
	// Initialize volk context
	volkInitialize();

	// Config application info
	VkApplicationInfo app_info{VK_STRUCTURE_TYPE_APPLICATION_INFO};

	uint32_t sdk_version = VK_HEADER_VERSION_COMPLETE;
	uint32_t api_version = 0;

	PFN_vkEnumerateInstanceVersion enumerate_instance_version = reinterpret_cast<PFN_vkEnumerateInstanceVersion>(vkGetInstanceProcAddr(nullptr, "vkEnumerateInstanceVersion"));

	if (enumerate_instance_version)
	{
		enumerate_instance_version(&api_version);
	}
	else
	{
		api_version = VK_VERSION_1_0;
	}

	if (sdk_version > api_version)
	{
		std::string sdk_version_str = std::to_string(VK_VERSION_MAJOR(sdk_version)) + "." + std::to_string(VK_VERSION_MINOR(sdk_version)) + "." + std::to_string(VK_VERSION_PATCH(sdk_version));
		std::string api_version_str = std::to_string(VK_VERSION_MAJOR(api_version)) + "." + std::to_string(VK_VERSION_MINOR(api_version)) + "." + std::to_string(VK_VERSION_PATCH(api_version));
		LOG_WARN("Using Vulkan {}, please upgrade your graphics driver to support Vulkan {}", api_version_str, sdk_version_str);
	}

	app_info.pApplicationName   = "IlumEngine";
	app_info.pEngineName        = "IlumEngine";
	app_info.engineVersion      = VK_MAKE_VERSION(0, 0, 1);
	app_info.applicationVersion = VK_MAKE_VERSION(0, 0, 1);
	app_info.apiVersion         = std::min(sdk_version, api_version);

	// Check out extensions support
	std::vector<const char *> extensions_support = GetInstanceExtensionSupported(s_instance_extensions);

	// Config instance info
	VkInstanceCreateInfo create_info{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
	create_info.pApplicationInfo        = &app_info;
	create_info.enabledExtensionCount   = static_cast<uint32_t>(extensions_support.size());
	create_info.ppEnabledExtensionNames = extensions_support.data();
	create_info.enabledLayerCount       = 0;

	// Enable validation layers
#ifdef _DEBUG
	// Validation features
	VkValidationFeaturesEXT validation_features{VK_STRUCTURE_TYPE_VALIDATION_FEATURES_EXT};
	validation_features.enabledValidationFeatureCount = static_cast<uint32_t>(s_validation_extensions.size());
	validation_features.pEnabledValidationFeatures    = s_validation_extensions.data();

	// Enable validation layer
	for (auto &layer : s_validation_layers)
	{
		if (CheckLayerSupported(layer))
		{
			LOG_INFO("Enable validation layer: {}", layer);
			create_info.enabledLayerCount   = static_cast<uint32_t>(s_validation_layers.size());
			create_info.ppEnabledLayerNames = s_validation_layers.data();
			create_info.pNext               = &validation_features;
			break;
		}
		else
		{
			LOG_ERROR("Validation layer was required, but not avaliable, disabling debugging");
		}
	}
#endif        // _DEBUG

	// Create instance
	if (vkCreateInstance(&create_info, nullptr, &m_instance) != VK_SUCCESS)
	{
		LOG_ERROR("Failed to create vulkan instance!");
		return;
	}
	else
	{
		// Config to volk
		volkLoadInstance(m_instance);
	}

	// Initialize instance extension functions
	vkCreateDebugUtilsMessengerEXT          = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(vkGetInstanceProcAddr(m_instance, "vkCreateDebugUtilsMessengerEXT"));
	vkDestroyDebugUtilsMessengerEXT         = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(vkGetInstanceProcAddr(m_instance, "vkDestroyDebugUtilsMessengerEXT"));
	vkSetDebugUtilsObjectTagEXT             = reinterpret_cast<PFN_vkSetDebugUtilsObjectTagEXT>(vkGetInstanceProcAddr(m_instance, "vkSetDebugUtilsObjectTagEXT"));
	vkSetDebugUtilsObjectNameEXT            = reinterpret_cast<PFN_vkSetDebugUtilsObjectNameEXT>(vkGetInstanceProcAddr(m_instance, "vkSetDebugUtilsObjectNameEXT"));
	vkCmdBeginDebugUtilsLabelEXT            = reinterpret_cast<PFN_vkCmdBeginDebugUtilsLabelEXT>(vkGetInstanceProcAddr(m_instance, "vkCmdBeginDebugUtilsLabelEXT"));
	vkCmdEndDebugUtilsLabelEXT              = reinterpret_cast<PFN_vkCmdEndDebugUtilsLabelEXT>(vkGetInstanceProcAddr(m_instance, "vkCmdEndDebugUtilsLabelEXT"));
	vkGetPhysicalDeviceMemoryProperties2KHR = reinterpret_cast<PFN_vkGetPhysicalDeviceMemoryProperties2KHR>(vkGetInstanceProcAddr(m_instance, "vkGetPhysicalRHIDeviceMemoryProperties2KHR"));

	// Enable debugger
#ifdef _DEBUG
	if (vkCreateDebugUtilsMessengerEXT)
	{
		VkDebugUtilsMessengerCreateInfoEXT create_info{VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT};
		create_info.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
		create_info.messageType     = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
		create_info.pfnUserCallback = ValidationCallback;

		vkCreateDebugUtilsMessengerEXT(m_instance, &create_info, nullptr, &vkDebugUtilsMessengerEXT);
	}
#endif        // _DEBUG
}

void RHIDevice::CreatePhysicalDevice()
{
	uint32_t physical_device_count = 0;
	vkEnumeratePhysicalDevices(m_instance, &physical_device_count, nullptr);

	// Get all physical devices
	std::vector<VkPhysicalDevice> physical_devices(physical_device_count);
	vkEnumeratePhysicalDevices(m_instance, &physical_device_count, physical_devices.data());

	// Select suitable physical device
	m_physical_device = SelectPhysicalDevice(physical_devices, s_device_extensions);

	m_depth_format = FindDepthFormat(m_physical_device);
}

void RHIDevice::CreateSurface()
{
	if (glfwCreateWindowSurface(m_instance, p_window->m_handle, nullptr, &m_surface) != VK_SUCCESS)
	{
		LOG_FATAL("Failed to create window surface!");
	}
}

void RHIDevice::CreateLogicalDevice()
{
	// Queue supporting
	uint32_t queue_family_property_count = 0;
	vkGetPhysicalDeviceQueueFamilyProperties(m_physical_device, &queue_family_property_count, nullptr);
	std::vector<VkQueueFamilyProperties> queue_family_properties(queue_family_property_count);
	vkGetPhysicalDeviceQueueFamilyProperties(m_physical_device, &queue_family_property_count, queue_family_properties.data());

	std::optional<uint32_t> graphics_family, compute_family, transfer_family, present_family;

	graphics_family = GetQueueFamilyIndex(queue_family_properties, VK_QUEUE_GRAPHICS_BIT);
	transfer_family = GetQueueFamilyIndex(queue_family_properties, VK_QUEUE_TRANSFER_BIT);
	compute_family  = GetQueueFamilyIndex(queue_family_properties, VK_QUEUE_COMPUTE_BIT);

	VkQueueFlags support_queues = 0;

	for (uint32_t i = 0; i < queue_family_property_count; i++)
	{
		// Check for presentation support
		VkBool32 present_support;
		vkGetPhysicalDeviceSurfaceSupportKHR(m_physical_device, i, m_surface, &present_support);

		if (queue_family_properties[i].queueCount > 0 && present_support)
		{
			present_family   = i;
			m_present_family = i;
			break;
		}
	}

	if (graphics_family.has_value())
	{
		m_graphics_family = graphics_family.value();
		support_queues |= VK_QUEUE_GRAPHICS_BIT;
	}

	if (compute_family.has_value())
	{
		m_compute_family = compute_family.value();
		support_queues |= VK_QUEUE_COMPUTE_BIT;
	}

	if (transfer_family.has_value())
	{
		m_transfer_family = transfer_family.value();
		support_queues |= VK_QUEUE_TRANSFER_BIT;
	}

	if (!graphics_family)
	{
		throw std::runtime_error("Failed to find queue graphics family support!");
	}

	// Create device queue
	std::vector<VkDeviceQueueCreateInfo> queue_create_infos;

	uint32_t max_count = 0;
	for (auto &queue_family_property : queue_family_properties)
	{
		max_count = max_count < queue_family_property.queueCount ? queue_family_property.queueCount : max_count;
	}

	std::vector<float> queue_priorities(max_count, 1.f);

	if (support_queues & VK_QUEUE_GRAPHICS_BIT)
	{
		VkDeviceQueueCreateInfo graphics_queue_create_info = {};
		graphics_queue_create_info.sType                   = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
		graphics_queue_create_info.queueFamilyIndex        = m_graphics_family;
		graphics_queue_create_info.queueCount              = queue_family_properties[m_graphics_family].queueCount;
		graphics_queue_create_info.pQueuePriorities        = queue_priorities.data();
		queue_create_infos.emplace_back(graphics_queue_create_info);
	}
	else
	{
		m_graphics_family = 0;
	}

	if (support_queues & VK_QUEUE_COMPUTE_BIT && m_compute_family != m_graphics_family)
	{
		VkDeviceQueueCreateInfo compute_queue_create_info = {};
		compute_queue_create_info.sType                   = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
		compute_queue_create_info.queueFamilyIndex        = m_compute_family;
		compute_queue_create_info.queueCount              = queue_family_properties[m_compute_family].queueCount;
		compute_queue_create_info.pQueuePriorities        = queue_priorities.data();
		queue_create_infos.emplace_back(compute_queue_create_info);
	}
	else
	{
		m_compute_family = m_graphics_family;
	}

	if (support_queues & VK_QUEUE_TRANSFER_BIT && m_transfer_family != m_graphics_family && m_transfer_family != m_compute_family)
	{
		VkDeviceQueueCreateInfo transfer_queue_create_info = {};
		transfer_queue_create_info.sType                   = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
		transfer_queue_create_info.queueFamilyIndex        = m_transfer_family;
		transfer_queue_create_info.queueCount              = queue_family_properties[m_transfer_family].queueCount;
		transfer_queue_create_info.pQueuePriorities        = queue_priorities.data();
		queue_create_infos.emplace_back(transfer_queue_create_info);
	}
	else
	{
		m_transfer_family = m_graphics_family;
	}

	// Enable logical device features
	VkPhysicalDeviceFeatures physical_device_features;
	vkGetPhysicalDeviceFeatures(m_physical_device, &physical_device_features);

#define ENABLE_DEVICE_FEATURE(feature)                              \
	if (physical_device_features.feature)                           \
	{                                                               \
		physical_device_features.feature = VK_TRUE;                 \
		LOG_INFO("Physical device feature enable: {}", #feature)    \
	}                                                               \
	else                                                            \
	{                                                               \
		LOG_WARN("Physical device feature not found: {}", #feature) \
	}

	ENABLE_DEVICE_FEATURE(sampleRateShading);
	ENABLE_DEVICE_FEATURE(fillModeNonSolid);
	ENABLE_DEVICE_FEATURE(wideLines);
	ENABLE_DEVICE_FEATURE(samplerAnisotropy);
	ENABLE_DEVICE_FEATURE(vertexPipelineStoresAndAtomics);
	ENABLE_DEVICE_FEATURE(fragmentStoresAndAtomics);
	ENABLE_DEVICE_FEATURE(shaderStorageImageExtendedFormats);
	ENABLE_DEVICE_FEATURE(shaderStorageImageWriteWithoutFormat);
	ENABLE_DEVICE_FEATURE(geometryShader);
	ENABLE_DEVICE_FEATURE(tessellationShader);
	ENABLE_DEVICE_FEATURE(multiViewport);
	ENABLE_DEVICE_FEATURE(imageCubeArray);
	ENABLE_DEVICE_FEATURE(robustBufferAccess);
	ENABLE_DEVICE_FEATURE(multiDrawIndirect);
	ENABLE_DEVICE_FEATURE(drawIndirectFirstInstance);

	// Enable Ray Tracing Extension
	VkPhysicalDeviceAccelerationStructureFeaturesKHR acceleration_structure_feature = {};
	acceleration_structure_feature.sType                                            = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR;
	acceleration_structure_feature.accelerationStructure                            = VK_TRUE;

	VkPhysicalDeviceRayTracingPipelineFeaturesKHR raty_tracing_pipeline_feature = {};
	raty_tracing_pipeline_feature.sType                                         = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR;
	raty_tracing_pipeline_feature.rayTracingPipeline                            = VK_TRUE;
	raty_tracing_pipeline_feature.pNext                                         = &acceleration_structure_feature;

	// Enable Mesh Shader Extension
	VkPhysicalDeviceMeshShaderFeaturesNV mesh_shader_feature = {};
	mesh_shader_feature.sType                                = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_NV;
	mesh_shader_feature.meshShader                           = VK_TRUE;
	mesh_shader_feature.taskShader                           = VK_TRUE;
	mesh_shader_feature.pNext                                = &raty_tracing_pipeline_feature;

	// Enable Vulkan 1.2 Features
	VkPhysicalDeviceVulkan12Features vulkan12_features          = {};
	vulkan12_features.sType                                     = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
	vulkan12_features.drawIndirectCount                         = VK_TRUE;
	vulkan12_features.shaderSampledImageArrayNonUniformIndexing = VK_TRUE;
	vulkan12_features.runtimeDescriptorArray                    = VK_TRUE;
	vulkan12_features.descriptorBindingVariableDescriptorCount  = VK_TRUE;
	vulkan12_features.descriptorBindingPartiallyBound           = VK_TRUE;
	vulkan12_features.bufferDeviceAddress                       = VK_TRUE;
	vulkan12_features.shaderOutputLayer                         = VK_TRUE;
	vulkan12_features.shaderOutputViewportIndex                 = VK_TRUE;
	vulkan12_features.pNext                                     = &mesh_shader_feature;

	// Get support extensions
	auto support_extensions = GetDeviceExtensionSupport(m_physical_device, s_device_extensions);

	// Create device
	VkDeviceCreateInfo device_create_info   = {};
	device_create_info.sType                = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
	device_create_info.queueCreateInfoCount = static_cast<uint32_t>(queue_create_infos.size());
	device_create_info.pQueueCreateInfos    = queue_create_infos.data();
	if (!s_validation_layers.empty())
	{
		device_create_info.enabledLayerCount   = static_cast<uint32_t>(s_validation_layers.size());
		device_create_info.ppEnabledLayerNames = s_validation_layers.data();
	}
	device_create_info.enabledExtensionCount   = static_cast<uint32_t>(support_extensions.size());
	device_create_info.ppEnabledExtensionNames = support_extensions.data();
	device_create_info.pEnabledFeatures        = &physical_device_features;
	device_create_info.pNext                   = &vulkan12_features;

	if (vkCreateDevice(m_physical_device, &device_create_info, nullptr, &m_device) != VK_SUCCESS)
	{
		LOG_ERROR("Failed to create logical device!");
		return;
	}

	// Volk load context
	volkLoadDevice(m_device);

	// Get device queues
	vkGetDeviceQueue(m_device, m_graphics_family, 1, &m_graphics_queue);
	vkGetDeviceQueue(m_device, m_compute_family, 1, &m_compute_queue);
	vkGetDeviceQueue(m_device, m_transfer_family, 1, &m_transfer_queue);
	vkGetDeviceQueue(m_device, m_present_family, 1, &m_present_queue);

	// Create Vma allocator
	VmaAllocatorCreateInfo allocator_info = {};
	allocator_info.physicalDevice         = m_physical_device;
	allocator_info.device                 = m_device;
	allocator_info.instance               = m_instance;
	allocator_info.vulkanApiVersion       = VK_API_VERSION_1_2;
	allocator_info.flags                  = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
	if (vmaCreateAllocator(&allocator_info, &m_allocator) != VK_SUCCESS)
	{
		LOG_FATAL("Failed to create vulkan memory allocator");
	}
}

void RHIDevice::CreateSwapchain()
{
	vkDeviceWaitIdle(m_device);

	m_swapchain_images.clear();

	// capabilities
	VkSurfaceCapabilitiesKHR capabilities;
	vkGetPhysicalDeviceSurfaceCapabilitiesKHR(m_physical_device, m_surface, &capabilities);

	// formats
	uint32_t                        format_count;
	std::vector<VkSurfaceFormatKHR> formats;
	vkGetPhysicalDeviceSurfaceFormatsKHR(m_physical_device, m_surface, &format_count, nullptr);
	if (format_count != 0)
	{
		formats.resize(format_count);
		vkGetPhysicalDeviceSurfaceFormatsKHR(m_physical_device, m_surface, &format_count, formats.data());
	}

	// present modes
	uint32_t                      presentmode_count;
	std::vector<VkPresentModeKHR> presentmodes;
	vkGetPhysicalDeviceSurfacePresentModesKHR(m_physical_device, m_surface, &presentmode_count, nullptr);
	if (presentmode_count != 0)
	{
		presentmodes.resize(presentmode_count);
		vkGetPhysicalDeviceSurfacePresentModesKHR(m_physical_device, m_surface, &presentmode_count, presentmodes.data());
	}

	// Choose swapchain surface format
	VkSurfaceFormatKHR chosen_surface_format = {};
	for (const auto &surface_format : formats)
	{
		if (surface_format.format == VK_FORMAT_B8G8R8A8_UNORM &&
		    surface_format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
		{
			chosen_surface_format = surface_format;
		}
	}
	if (chosen_surface_format.format == VK_FORMAT_UNDEFINED)
	{
		chosen_surface_format = formats[0];
	}

	// Choose swapchain present mode
	VkPresentModeKHR chosen_presentMode = VK_PRESENT_MODE_MAX_ENUM_KHR;
	for (VkPresentModeKHR present_mode : presentmodes)
	{
		if (VK_PRESENT_MODE_MAILBOX_KHR == present_mode)
		{
			chosen_presentMode = VK_PRESENT_MODE_MAILBOX_KHR;
		}
	}
	if (chosen_presentMode == VK_PRESENT_MODE_MAX_ENUM_KHR)
	{
		chosen_presentMode = VK_PRESENT_MODE_FIFO_KHR;
	}

	// Choose swapchain extent
	VkExtent2D chosen_extent = {};
	if (capabilities.currentExtent.width != UINT32_MAX)
	{
		chosen_extent = capabilities.currentExtent;
	}
	else
	{
		int32_t width, height;
		glfwGetFramebufferSize(p_window->m_handle, &width, &height);

		VkExtent2D actualExtent = {static_cast<uint32_t>(width), static_cast<uint32_t>(height)};

		actualExtent.width =
		    std::clamp(actualExtent.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
		actualExtent.height =
		    std::clamp(actualExtent.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);

		chosen_extent = actualExtent;
	}

	uint32_t image_count = capabilities.minImageCount + 1;
	if (capabilities.maxImageCount > 0 && image_count > capabilities.maxImageCount)
	{
		image_count = capabilities.maxImageCount;
	}

	VkSwapchainCreateInfoKHR createInfo = {};
	createInfo.sType                    = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
	createInfo.surface                  = m_surface;

	createInfo.minImageCount    = image_count;
	createInfo.imageFormat      = chosen_surface_format.format;
	createInfo.imageColorSpace  = chosen_surface_format.colorSpace;
	createInfo.imageExtent      = chosen_extent;
	createInfo.imageArrayLayers = 1;
	createInfo.imageUsage       = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

	uint32_t queueFamilyIndices[] = {m_graphics_family, m_present_family};

	if (m_graphics_family != m_graphics_family)
	{
		createInfo.imageSharingMode      = VK_SHARING_MODE_CONCURRENT;
		createInfo.queueFamilyIndexCount = 2;
		createInfo.pQueueFamilyIndices   = queueFamilyIndices;
	}
	else
	{
		createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
	}

	createInfo.preTransform   = capabilities.currentTransform;
	createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
	createInfo.presentMode    = chosen_presentMode;
	createInfo.clipped        = VK_TRUE;

	createInfo.oldSwapchain = m_swapchain;

	if (vkCreateSwapchainKHR(m_device, &createInfo, nullptr, &m_swapchain) != VK_SUCCESS)
	{
		LOG_FATAL("Failed to create swapchain!");
	}

	std::vector<VkImage> images;
	vkGetSwapchainImagesKHR(m_device, m_swapchain, &image_count, nullptr);
	images.resize(image_count);
	vkGetSwapchainImagesKHR(m_device, m_swapchain, &image_count, images.data());

	TextureDesc swapchain_image_desc;
	swapchain_image_desc.depth  = 1;
	swapchain_image_desc.width  = chosen_extent.width;
	swapchain_image_desc.height = chosen_extent.height;
	swapchain_image_desc.format = chosen_surface_format.format;
	swapchain_image_desc.usage  = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

	for (size_t i = 0; i < images.size(); i++)
	{
		m_swapchain_images.emplace_back(std::make_unique<Texture>(this, swapchain_image_desc, images[i]));
		m_swapchain_images.back()->SetName(std::string("Swapchain Image #") + std::to_string(i));
	}

	m_swapchain_format = chosen_surface_format.format;

	if (m_frames.empty())
	{
		for (uint32_t i = 0; i < images.size(); i++)
		{
			m_frames.emplace_back(std::make_unique<Frame>(this));
		}
	}

	if (m_present_complete.empty())
	{
		for (uint32_t i = 0; i < images.size(); i++)
		{
			m_present_complete.push_back(m_frames[i]->AllocateSemaphore());
		}
	}

	if (m_render_complete.empty())
	{
		for (uint32_t i = 0; i < images.size(); i++)
		{
			m_render_complete.push_back(m_frames[i]->AllocateSemaphore());
		}
	}

	m_current_frame = 0;
}

void RHIDevice::CreateAllocator()
{
	m_shader_allocator     = std::make_unique<ShaderAllocator>(this);
	m_descriptor_allocator = std::make_unique<DescriptorAllocator>(this);
}

VkPipeline RHIDevice::AllocateGraphicsPipeline(const PipelineState &pso, VkRenderPass render_pass)
{
	size_t hash = pso.Hash();

	if (m_pipelines.find(hash) != m_pipelines.end())
	{
		return m_pipelines[hash];
	}

	m_pipelines[hash] = VK_NULL_HANDLE;

	ASSERT(render_pass);

	// Input Assembly State
	VkPipelineInputAssemblyStateCreateInfo input_assembly_state_create_info = {};
	input_assembly_state_create_info.sType                                  = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
	input_assembly_state_create_info.topology                               = pso.GetInputAssemblyState().topology;
	input_assembly_state_create_info.flags                                  = 0;
	input_assembly_state_create_info.primitiveRestartEnable                 = pso.GetInputAssemblyState().primitive_restart_enable;

	// Rasterization State
	VkPipelineRasterizationStateCreateInfo rasterization_state_create_info = {};
	rasterization_state_create_info.sType                                  = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
	rasterization_state_create_info.polygonMode                            = pso.GetRasterizationState().polygon_mode;
	rasterization_state_create_info.cullMode                               = pso.GetRasterizationState().cull_mode;
	rasterization_state_create_info.frontFace                              = pso.GetRasterizationState().front_face;
	rasterization_state_create_info.flags                                  = 0;
	rasterization_state_create_info.depthBiasEnable                        = VK_TRUE;
	rasterization_state_create_info.lineWidth                              = 1.0f;

	// Color Blend Attachment State
	std::vector<VkPipelineColorBlendAttachmentState> color_blend_attachment_states(pso.GetColorBlendState().attachment_states.size());

	for (uint32_t i = 0; i < color_blend_attachment_states.size(); i++)
	{
		color_blend_attachment_states[i].blendEnable         = pso.GetColorBlendState().attachment_states[i].blend_enable;
		color_blend_attachment_states[i].srcColorBlendFactor = pso.GetColorBlendState().attachment_states[i].src_color_blend_factor;
		color_blend_attachment_states[i].dstColorBlendFactor = pso.GetColorBlendState().attachment_states[i].dst_color_blend_factor;
		color_blend_attachment_states[i].colorBlendOp        = pso.GetColorBlendState().attachment_states[i].color_blend_op;
		color_blend_attachment_states[i].srcAlphaBlendFactor = pso.GetColorBlendState().attachment_states[i].src_alpha_blend_factor;
		color_blend_attachment_states[i].dstAlphaBlendFactor = pso.GetColorBlendState().attachment_states[i].dst_alpha_blend_factor;
		color_blend_attachment_states[i].alphaBlendOp        = pso.GetColorBlendState().attachment_states[i].alpha_blend_op;
		color_blend_attachment_states[i].colorWriteMask      = pso.GetColorBlendState().attachment_states[i].color_write_mask;
	}

	VkPipelineColorBlendStateCreateInfo color_blend_state_create_info = {};
	color_blend_state_create_info.sType                               = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
	color_blend_state_create_info.logicOpEnable                       = VK_FALSE;
	color_blend_state_create_info.logicOp                             = VK_LOGIC_OP_COPY;
	color_blend_state_create_info.attachmentCount                     = static_cast<uint32_t>(color_blend_attachment_states.size());
	color_blend_state_create_info.pAttachments                        = color_blend_attachment_states.data();
	color_blend_state_create_info.blendConstants[0]                   = 0.0f;
	color_blend_state_create_info.blendConstants[1]                   = 0.0f;
	color_blend_state_create_info.blendConstants[2]                   = 0.0f;
	color_blend_state_create_info.blendConstants[3]                   = 0.0f;

	// Depth Stencil State
	VkPipelineDepthStencilStateCreateInfo depth_stencil_state_create_info = {};
	depth_stencil_state_create_info.sType                                 = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
	depth_stencil_state_create_info.depthTestEnable                       = pso.GetDepthStencilState().depth_test_enable;
	depth_stencil_state_create_info.depthWriteEnable                      = pso.GetDepthStencilState().depth_write_enable;
	depth_stencil_state_create_info.depthCompareOp                        = pso.GetDepthStencilState().depth_compare_op;
	depth_stencil_state_create_info.back                                  = pso.GetDepthStencilState().back;
	depth_stencil_state_create_info.front                                 = pso.GetDepthStencilState().front;
	depth_stencil_state_create_info.stencilTestEnable                     = pso.GetDepthStencilState().stencil_test_enable;

	// Viewport State
	VkPipelineViewportStateCreateInfo viewport_state_create_info = {};
	viewport_state_create_info.sType                             = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
	viewport_state_create_info.viewportCount                     = pso.GetViewportState().viewport_count;
	viewport_state_create_info.scissorCount                      = pso.GetViewportState().scissor_count;
	viewport_state_create_info.flags                             = 0;

	// Multisample State
	VkPipelineMultisampleStateCreateInfo multisample_state_create_info = {};
	multisample_state_create_info.sType                                = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
	multisample_state_create_info.rasterizationSamples                 = pso.GetMultisampleState().sample_count;
	multisample_state_create_info.flags                                = 0;

	// Dynamic State
	VkPipelineDynamicStateCreateInfo dynamic_state_create_info = {};
	dynamic_state_create_info.sType                            = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
	dynamic_state_create_info.pDynamicStates                   = pso.GetDynamicState().dynamic_states.data();
	dynamic_state_create_info.dynamicStateCount                = static_cast<uint32_t>(pso.GetDynamicState().dynamic_states.size());
	dynamic_state_create_info.flags                            = 0;

	std::vector<VkPipelineShaderStageCreateInfo> pipeline_shader_stage_create_infos;
	for (auto &shader_desc : pso.GetShaders())
	{
		VkPipelineShaderStageCreateInfo shader_stage_create_info = {};

		shader_stage_create_info.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		shader_stage_create_info.stage  = shader_desc.stage;
		shader_stage_create_info.module = LoadShader(shader_desc);
		shader_stage_create_info.pName  = shader_desc.entry_point.c_str();
		pipeline_shader_stage_create_infos.push_back(shader_stage_create_info);
	}

	// Vertex Input State
	VkPipelineVertexInputStateCreateInfo vertex_input_state_create_info = {};
	vertex_input_state_create_info.sType                                = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
	vertex_input_state_create_info.vertexAttributeDescriptionCount      = static_cast<uint32_t>(pso.GetVertexInputState().attribute_descriptions.size());
	vertex_input_state_create_info.pVertexAttributeDescriptions         = pso.GetVertexInputState().attribute_descriptions.data();
	vertex_input_state_create_info.vertexBindingDescriptionCount        = static_cast<uint32_t>(pso.GetVertexInputState().binding_descriptions.size());
	vertex_input_state_create_info.pVertexBindingDescriptions           = pso.GetVertexInputState().binding_descriptions.data();

	VkGraphicsPipelineCreateInfo graphics_pipeline_create_info = {};
	graphics_pipeline_create_info.sType                        = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
	graphics_pipeline_create_info.stageCount                   = static_cast<uint32_t>(pipeline_shader_stage_create_infos.size());
	graphics_pipeline_create_info.pStages                      = pipeline_shader_stage_create_infos.data();

	graphics_pipeline_create_info.pInputAssemblyState = &input_assembly_state_create_info;
	graphics_pipeline_create_info.pRasterizationState = &rasterization_state_create_info;
	graphics_pipeline_create_info.pColorBlendState    = &color_blend_state_create_info;
	graphics_pipeline_create_info.pDepthStencilState  = &depth_stencil_state_create_info;
	graphics_pipeline_create_info.pViewportState      = &viewport_state_create_info;
	graphics_pipeline_create_info.pMultisampleState   = &multisample_state_create_info;
	graphics_pipeline_create_info.pDynamicState       = &dynamic_state_create_info;
	graphics_pipeline_create_info.pVertexInputState   = &vertex_input_state_create_info;

	graphics_pipeline_create_info.layout             = AllocatePipelineLayout(pso);
	graphics_pipeline_create_info.renderPass         = render_pass;
	graphics_pipeline_create_info.subpass            = 0;
	graphics_pipeline_create_info.basePipelineHandle = VK_NULL_HANDLE;
	graphics_pipeline_create_info.basePipelineIndex  = -1;

	vkCreateGraphicsPipelines(m_device, m_pipeline_cache, 1, &graphics_pipeline_create_info, nullptr, &m_pipelines[hash]);

	return m_pipelines[hash];
}

VkPipeline RHIDevice::AllocateComputePipeline(const PipelineState &pso)
{
	size_t hash = pso.Hash();

	if (m_pipelines.find(hash) != m_pipelines.end())
	{
		return m_pipelines[hash];
	}

	m_pipelines[hash] = VK_NULL_HANDLE;

	VkPipelineShaderStageCreateInfo shader_stage_create_info = {};
	shader_stage_create_info.sType                           = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	shader_stage_create_info.stage                           = VK_SHADER_STAGE_COMPUTE_BIT;
	shader_stage_create_info.module                          = LoadShader(pso.GetShaders()[0]);
	shader_stage_create_info.pName                           = pso.GetShaders()[0].entry_point.c_str();

	VkComputePipelineCreateInfo compute_pipeline_create_info = {};
	compute_pipeline_create_info.sType                       = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
	compute_pipeline_create_info.stage                       = shader_stage_create_info;
	compute_pipeline_create_info.layout                      = AllocatePipelineLayout(pso);
	compute_pipeline_create_info.basePipelineIndex           = 0;
	compute_pipeline_create_info.basePipelineHandle          = VK_NULL_HANDLE;

	vkCreateComputePipelines(m_device, m_pipeline_cache, 1, &compute_pipeline_create_info, nullptr, &m_pipelines[hash]);

	return m_pipelines[hash];
}

VkPipeline RHIDevice::AllocateRayTracingPipeline(const PipelineState &pso)
{
	size_t hash = pso.Hash();

	if (m_pipelines.find(hash) != m_pipelines.end())
	{
		return m_pipelines[hash];
	}

	m_pipelines[hash] = VK_NULL_HANDLE;

	std::vector<VkRayTracingShaderGroupCreateInfoKHR> shader_group_create_infos;
	std::vector<VkPipelineShaderStageCreateInfo>      pipeline_shader_stage_create_infos;

	uint32_t raygen_count   = 0;
	uint32_t raymiss_count  = 0;
	uint32_t rayhit_count   = 0;
	uint32_t callable_count = 0;

	// Ray Generation Group
	{
		for (auto &shader : pso.GetShaders())
		{
			if (shader.stage == VK_SHADER_STAGE_RAYGEN_BIT_KHR)
			{
				VkPipelineShaderStageCreateInfo pipeline_shader_stage_create_info = {};
				pipeline_shader_stage_create_info.sType                           = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
				pipeline_shader_stage_create_info.stage                           = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
				pipeline_shader_stage_create_info.module                          = LoadShader(shader);
				pipeline_shader_stage_create_info.pName                           = shader.entry_point.c_str();
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
		for (auto &shader : pso.GetShaders())
		{
			if (shader.stage == VK_SHADER_STAGE_MISS_BIT_KHR)
			{
				VkPipelineShaderStageCreateInfo pipeline_shader_stage_create_info = {};
				pipeline_shader_stage_create_info.sType                           = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
				pipeline_shader_stage_create_info.stage                           = VK_SHADER_STAGE_MISS_BIT_KHR;
				pipeline_shader_stage_create_info.module                          = LoadShader(shader);
				pipeline_shader_stage_create_info.pName                           = shader.entry_point.c_str();
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
		for (auto &shader : pso.GetShaders())
		{
			if (shader.stage == VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR)
			{
				VkPipelineShaderStageCreateInfo pipeline_shader_stage_create_info = {};
				pipeline_shader_stage_create_info.sType                           = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
				pipeline_shader_stage_create_info.stage                           = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
				pipeline_shader_stage_create_info.module                          = LoadShader(shader);
				pipeline_shader_stage_create_info.pName                           = shader.entry_point.c_str();
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
		for (auto &shader : pso.GetShaders())
		{
			if (shader.stage == VK_SHADER_STAGE_ANY_HIT_BIT_KHR)
			{
				VkPipelineShaderStageCreateInfo pipeline_shader_stage_create_info = {};
				pipeline_shader_stage_create_info.sType                           = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
				pipeline_shader_stage_create_info.stage                           = VK_SHADER_STAGE_ANY_HIT_BIT_KHR;
				pipeline_shader_stage_create_info.module                          = LoadShader(shader);
				pipeline_shader_stage_create_info.pName                           = shader.entry_point.c_str();
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
		for (auto &shader : pso.GetShaders())
		{
			if (shader.stage == VK_SHADER_STAGE_INTERSECTION_BIT_KHR)
			{
				VkPipelineShaderStageCreateInfo pipeline_shader_stage_create_info = {};
				pipeline_shader_stage_create_info.sType                           = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
				pipeline_shader_stage_create_info.stage                           = VK_SHADER_STAGE_INTERSECTION_BIT_KHR;
				pipeline_shader_stage_create_info.module                          = LoadShader(shader);
				pipeline_shader_stage_create_info.pName                           = shader.entry_point.c_str();
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
		for (auto &shader : pso.GetShaders())
		{
			if (shader.stage == VK_SHADER_STAGE_CALLABLE_BIT_KHR)
			{
				VkPipelineShaderStageCreateInfo pipeline_shader_stage_create_info = {};
				pipeline_shader_stage_create_info.sType                           = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
				pipeline_shader_stage_create_info.stage                           = VK_SHADER_STAGE_CALLABLE_BIT_KHR;
				pipeline_shader_stage_create_info.module                          = LoadShader(shader);
				pipeline_shader_stage_create_info.pName                           = shader.entry_point.c_str();
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
	raytracing_pipeline_create_info.layout                            = AllocatePipelineLayout(pso);

	vkCreateRayTracingPipelinesKHR(m_device, VK_NULL_HANDLE, m_pipeline_cache, 1, &raytracing_pipeline_create_info, nullptr, &m_pipelines[hash]);

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

	m_shader_binding_tables[hash] = std::make_unique<ShaderBindingTable>();

	VkPhysicalDeviceRayTracingPipelinePropertiesKHR raytracing_properties = {};
	raytracing_properties.sType                                           = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR;
	VkPhysicalDeviceProperties2 deviceProperties2                         = {};
	deviceProperties2.sType                                               = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
	deviceProperties2.pNext                                               = &raytracing_properties;
	vkGetPhysicalDeviceProperties2(m_physical_device, &deviceProperties2);

	const uint32_t handle_size         = raytracing_properties.shaderGroupHandleSize;
	const uint32_t handle_size_aligned = (raytracing_properties.shaderGroupHandleSize + raytracing_properties.shaderGroupHandleAlignment - 1) & ~(raytracing_properties.shaderGroupHandleAlignment - 1);
	const uint32_t group_count         = static_cast<uint32_t>(shader_group_create_infos.size());
	const uint32_t sbt_size            = group_count * handle_size_aligned;

	std::vector<uint8_t> shader_handle_storage(sbt_size);

	auto result = vkGetRayTracingShaderGroupHandlesKHR(m_device, m_pipelines[hash], 0, group_count, sbt_size, shader_handle_storage.data());

	uint32_t handle_offset = 0;

	// Gen Group
	{
		m_shader_binding_tables[hash]->raygen = std::make_unique<ShaderBindingTableInfo>(this, raygen_count);
		std::memcpy(m_shader_binding_tables[hash]->raygen->GetData(), shader_handle_storage.data() + handle_offset, handle_size * raygen_count);
		handle_offset += raygen_count * handle_size_aligned;
	}

	// Miss Group
	{
		m_shader_binding_tables[hash]->miss = std::make_unique<ShaderBindingTableInfo>(this, raymiss_count);
		std::memcpy(m_shader_binding_tables[hash]->miss->GetData(), shader_handle_storage.data() + handle_offset, handle_size * raymiss_count);
		handle_offset += raymiss_count * handle_size_aligned;
	}

	// Hit Group
	{
		m_shader_binding_tables[hash]->hit = std::make_unique<ShaderBindingTableInfo>(this, rayhit_count);
		std::memcpy(m_shader_binding_tables[hash]->hit->GetData(), shader_handle_storage.data() + handle_offset, handle_size * rayhit_count);
		handle_offset += rayhit_count * handle_size_aligned;
	}

	// Callable Group
	{
		m_shader_binding_tables[hash]->callable = std::make_unique<ShaderBindingTableInfo>(this, callable_count);
		std::memcpy(m_shader_binding_tables[hash]->callable->GetData(), shader_handle_storage.data() + handle_offset, handle_size * callable_count);
		handle_offset += callable_count * handle_size_aligned;
	}

	return m_pipelines[hash];
}
}        // namespace Ilum