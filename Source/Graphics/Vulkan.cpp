#define VMA_IMPLEMENTATION

#include "Vulkan.hpp"

#include <cassert>
#include <stdexcept>
#include <string>

namespace Ilum::Graphics
{
#ifdef _DEBUG
const std::vector<const char *>                 InstanceExtension::extensions            = {"VK_KHR_surface", "VK_KHR_win32_surface", "VK_EXT_debug_report", "VK_EXT_debug_utils"};
const std::vector<const char *>                 InstanceExtension::validation_layers     = {"VK_LAYER_KHRONOS_validation"};
const std::vector<VkValidationFeatureEnableEXT> InstanceExtension::validation_extensions = {
    VK_VALIDATION_FEATURE_ENABLE_GPU_ASSISTED_RESERVE_BINDING_SLOT_EXT,
    VK_VALIDATION_FEATURE_ENABLE_GPU_ASSISTED_EXT,
    VK_VALIDATION_FEATURE_ENABLE_SYNCHRONIZATION_VALIDATION_EXT};
#else
const std::vector<const char *>                 InstanceExtension::extensions            = {"VK_KHR_surface", "VK_KHR_win32_surface"};
const std::vector<const char *>                 InstanceExtension::validation_layers     = {};
const std::vector<VkValidationFeatureEnableEXT> InstanceExtension::validation_extensions = {};
#endif        // _DEBUG

PFN_vkCreateDebugUtilsMessengerEXT          InstanceFunctionEXT::CreateDebugUtilsMessengerEXT          = nullptr;
VkDebugUtilsMessengerEXT                    InstanceFunctionEXT::DebugUtilsMessengerEXT                = nullptr;
PFN_vkDestroyDebugUtilsMessengerEXT         InstanceFunctionEXT::DestroyDebugUtilsMessengerEXT         = nullptr;
PFN_vkSetDebugUtilsObjectTagEXT             InstanceFunctionEXT::SetDebugUtilsObjectTagEXT             = nullptr;
PFN_vkSetDebugUtilsObjectNameEXT            InstanceFunctionEXT::SetDebugUtilsObjectNameEXT            = nullptr;
PFN_vkCmdBeginDebugUtilsLabelEXT            InstanceFunctionEXT::CmdBeginDebugUtilsLabelEXT            = nullptr;
PFN_vkCmdEndDebugUtilsLabelEXT              InstanceFunctionEXT::CmdEndDebugUtilsLabelEXT              = nullptr;
PFN_vkGetPhysicalDeviceMemoryProperties2KHR InstanceFunctionEXT::GetPhysicalDeviceMemoryProperties2KHR = nullptr;

const std::vector<const char *> DeviceExtension::extensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME,
    VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
    VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME,
    VK_KHR_RAY_QUERY_EXTENSION_NAME,
    VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
    VK_KHR_SHADER_DRAW_PARAMETERS_EXTENSION_NAME};

static inline VKAPI_ATTR VkBool32 VKAPI_CALL Callback(VkDebugUtilsMessageSeverityFlagBitsEXT msg_severity, VkDebugUtilsMessageTypeFlagsEXT msg_type, const VkDebugUtilsMessengerCallbackDataEXT *callback_data, void *user_data)
{
	if (msg_severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT)
	{
		VK_INFO(callback_data->pMessage);
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

const bool vk_check(VkResult result)
{
	vk_assert(result);

	return result == VK_SUCCESS;
}

void vk_assert(VkResult result)
{
	//#ifdef _DEBUG
	assert(result == VK_SUCCESS);
	//#else
	if (result != VK_SUCCESS)
	{
		LOG_ERROR("{}", std::to_string(result));

		throw std::runtime_error(std::to_string(result));
	}
	//#endif        // _DEBUG
}

std::string shader_stage_to_string(VkShaderStageFlags stage)
{
	std::string result = "";
#define ADD_SHADER_STAGE(vk_shader_stage)          \
	if (stage & vk_shader_stage)                   \
	{                                              \
		if (!result.empty())                       \
		{                                          \
			result += " | ";                       \
		}                                          \
		result += std::to_string(vk_shader_stage); \
	}

	ADD_SHADER_STAGE(VK_SHADER_STAGE_VERTEX_BIT);
	ADD_SHADER_STAGE(VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT);
	ADD_SHADER_STAGE(VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT);
	ADD_SHADER_STAGE(VK_SHADER_STAGE_GEOMETRY_BIT);
	ADD_SHADER_STAGE(VK_SHADER_STAGE_FRAGMENT_BIT);
	ADD_SHADER_STAGE(VK_SHADER_STAGE_COMPUTE_BIT);
	ADD_SHADER_STAGE(VK_SHADER_STAGE_RAYGEN_BIT_KHR);
	ADD_SHADER_STAGE(VK_SHADER_STAGE_ANY_HIT_BIT_KHR);
	ADD_SHADER_STAGE(VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR);
	ADD_SHADER_STAGE(VK_SHADER_STAGE_MISS_BIT_KHR);
	ADD_SHADER_STAGE(VK_SHADER_STAGE_INTERSECTION_BIT_KHR);
	ADD_SHADER_STAGE(VK_SHADER_STAGE_CALLABLE_BIT_KHR);
	ADD_SHADER_STAGE(VK_SHADER_STAGE_TASK_BIT_NV);
	ADD_SHADER_STAGE(VK_SHADER_STAGE_MESH_BIT_NV);

	if (result.empty())
	{
		ADD_SHADER_STAGE(VK_SHADER_STAGE_ALL_GRAPHICS);
		ADD_SHADER_STAGE(VK_SHADER_STAGE_ALL);
	}

	return result;
}

bool IsDepth(VkFormat format)
{
	switch (format)
	{
		case VK_FORMAT_D16_UNORM:
		case VK_FORMAT_D32_SFLOAT:
		case VK_FORMAT_D16_UNORM_S8_UINT:
		case VK_FORMAT_D24_UNORM_S8_UINT:
		case VK_FORMAT_D32_SFLOAT_S8_UINT:
			return true;
		default:
			return false;
	}
}

bool IsStencil(VkFormat format)
{
	switch (format)
	{
		case VK_FORMAT_S8_UINT:
		case VK_FORMAT_D16_UNORM_S8_UINT:
		case VK_FORMAT_D24_UNORM_S8_UINT:
		case VK_FORMAT_D32_SFLOAT_S8_UINT:
			return true;
		default:
			return false;
	}
}

bool IsDepthStencil(VkFormat format)
{
	return IsDepth(format) && IsStencil(format);
}

void VKDebugger::Initialize(VkInstance instance)
{
	if (InstanceFunctionEXT::CreateDebugUtilsMessengerEXT)
	{
		VkDebugUtilsMessengerCreateInfoEXT create_info{VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT};
		create_info.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
		create_info.messageType     = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
		create_info.pfnUserCallback = Callback;

		InstanceFunctionEXT::CreateDebugUtilsMessengerEXT(instance, &create_info, nullptr, &InstanceFunctionEXT::DebugUtilsMessengerEXT);
	}
}

void VKDebugger::Shutdown(VkInstance instance)
{
	if (!InstanceFunctionEXT::DestroyDebugUtilsMessengerEXT)
	{
		return;
	}

	InstanceFunctionEXT::DestroyDebugUtilsMessengerEXT(instance, InstanceFunctionEXT::DebugUtilsMessengerEXT, nullptr);
}

void VKDebugger::SetObjectName(VkDevice device, uint64_t object, VkObjectType object_type, const char *name)
{
	if (!InstanceFunctionEXT::SetDebugUtilsObjectNameEXT)
	{
		return;
	}

	VkDebugUtilsObjectNameInfoEXT name_info{VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT};
	name_info.pNext        = nullptr;
	name_info.objectType   = object_type;
	name_info.objectHandle = object;
	name_info.pObjectName  = name;
	InstanceFunctionEXT::SetDebugUtilsObjectNameEXT(device, &name_info);
}

void VKDebugger::SetObjectTag(VkDevice device, uint64_t object, VkObjectType object_type, uint64_t tag_name, size_t tag_size, const void *tag)
{
	if (!InstanceFunctionEXT::SetDebugUtilsObjectTagEXT)
	{
		return;
	}

	VkDebugUtilsObjectTagInfoEXT tag_info{VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_TAG_INFO_EXT};
	tag_info.pNext        = nullptr;
	tag_info.objectType   = object_type;
	tag_info.objectHandle = object;
	tag_info.tagName      = tag_name;
	tag_info.tagSize      = tag_size;
	tag_info.pTag         = tag;

	InstanceFunctionEXT::SetDebugUtilsObjectTagEXT(device, &tag_info);
}

void VKDebugger::MarkerBegin(VkCommandBuffer cmd_buffer, const char *name, const float r, const float g, const float b, const float a)
{
	if (!InstanceFunctionEXT::CmdBeginDebugUtilsLabelEXT)
	{
		return;
	}

	VkDebugUtilsLabelEXT label{VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT};
	label.pNext      = nullptr;
	label.pLabelName = name;
	label.color[0]   = r;
	label.color[1]   = g;
	label.color[2]   = b;
	label.color[3]   = a;
	InstanceFunctionEXT::CmdBeginDebugUtilsLabelEXT(cmd_buffer, &label);
}

void VKDebugger::MarkerEnd(VkCommandBuffer cmd_buffer)
{
	if (!InstanceFunctionEXT::CmdEndDebugUtilsLabelEXT)
	{
		return;
	}

	InstanceFunctionEXT::CmdEndDebugUtilsLabelEXT(cmd_buffer);
}

void VKDebugger::SetName(VkDevice device, VkCommandPool cmd_pool, const char *name)
{
	SetObjectName(device, (uint64_t) cmd_pool, VK_OBJECT_TYPE_COMMAND_POOL, name);
}

void VKDebugger::SetName(VkDevice device, VkCommandBuffer cmd_buffer, const char *name)
{
	SetObjectName(device, (uint64_t) cmd_buffer, VK_OBJECT_TYPE_COMMAND_BUFFER, name);
}

void VKDebugger::SetName(VkDevice device, VkQueue queue, const char *name)
{
	SetObjectName(device, (uint64_t) queue, VK_OBJECT_TYPE_QUEUE, name);
}

void VKDebugger::SetName(VkDevice device, VkImage image, const char *name)
{
	SetObjectName(device, (uint64_t) image, VK_OBJECT_TYPE_IMAGE, name);
}

void VKDebugger::SetName(VkDevice device, VkImageView image_view, const char *name)
{
	SetObjectName(device, (uint64_t) image_view, VK_OBJECT_TYPE_IMAGE_VIEW, name);
}

void VKDebugger::SetName(VkDevice device, VkSampler sampler, const char *name)
{
	SetObjectName(device, (uint64_t) sampler, VK_OBJECT_TYPE_SAMPLER, name);
}

void VKDebugger::SetName(VkDevice device, VkBuffer buffer, const char *name)
{
	SetObjectName(device, (uint64_t) buffer, VK_OBJECT_TYPE_BUFFER, name);
}

void VKDebugger::SetName(VkDevice device, VkBufferView buffer_view, const char *name)
{
	SetObjectName(device, (uint64_t) buffer_view, VK_OBJECT_TYPE_BUFFER_VIEW, name);
}

void VKDebugger::SetName(VkDevice device, VkDeviceMemory memory, const char *name)
{
	SetObjectName(device, (uint64_t) memory, VK_OBJECT_TYPE_DEVICE_MEMORY, name);
}

void VKDebugger::SetName(VkDevice device, VkAccelerationStructureKHR acceleration_structure, const char *name)
{
	SetObjectName(device, (uint64_t) acceleration_structure, VK_OBJECT_TYPE_ACCELERATION_STRUCTURE_KHR, name);
}

void VKDebugger::SetName(VkDevice device, VkShaderModule shader_module, const char *name)
{
	SetObjectName(device, (uint64_t) shader_module, VK_OBJECT_TYPE_SHADER_MODULE, name);
}

void VKDebugger::SetName(VkDevice device, VkPipeline pipeline, const char *name)
{
	SetObjectName(device, (uint64_t) pipeline, VK_OBJECT_TYPE_PIPELINE, name);
}

void VKDebugger::SetName(VkDevice device, VkPipelineLayout pipeline_layout, const char *name)
{
	SetObjectName(device, (uint64_t) pipeline_layout, VK_OBJECT_TYPE_PIPELINE_LAYOUT, name);
}

void VKDebugger::SetName(VkDevice device, VkRenderPass render_pass, const char *name)
{
	SetObjectName(device, (uint64_t) render_pass, VK_OBJECT_TYPE_RENDER_PASS, name);
}

void VKDebugger::SetName(VkDevice device, VkFramebuffer frame_buffer, const char *name)
{
	SetObjectName(device, (uint64_t) frame_buffer, VK_OBJECT_TYPE_FRAMEBUFFER, name);
}

void VKDebugger::SetName(VkDevice device, VkDescriptorSetLayout descriptor_set_layout, const char *name)
{
	SetObjectName(device, (uint64_t) descriptor_set_layout, VK_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT, name);
}

void VKDebugger::SetName(VkDevice device, VkDescriptorSet descriptor_set, const char *name)
{
	SetObjectName(device, (uint64_t) descriptor_set, VK_OBJECT_TYPE_DESCRIPTOR_SET, name);
}

void VKDebugger::SetName(VkDevice device, VkDescriptorPool descriptor_pool, const char *name)
{
	SetObjectName(device, (uint64_t) descriptor_pool, VK_OBJECT_TYPE_DESCRIPTOR_POOL, name);
}

void VKDebugger::SetName(VkDevice device, VkSemaphore semaphore, const char *name)
{
	SetObjectName(device, (uint64_t) semaphore, VK_OBJECT_TYPE_SEMAPHORE, name);
}

void VKDebugger::SetName(VkDevice device, VkFence fence, const char *name)
{
	SetObjectName(device, (uint64_t) fence, VK_OBJECT_TYPE_FENCE, name);
}

void VKDebugger::SetName(VkDevice device, VkEvent event, const char *name)
{
	SetObjectName(device, (uint64_t) event, VK_OBJECT_TYPE_EVENT, name);
}
}        // namespace Ilum::Graphics

std::string std::to_string(VkResult result)
{
	switch (result)
	{
		case VK_SUCCESS:
			return "VK_SUCCESS";
		case VK_NOT_READY:
			return "VK_NOT_READY";
		case VK_TIMEOUT:
			return "VK_TIMEOUT";
		case VK_EVENT_SET:
			return "VK_EVENT_SET";
		case VK_EVENT_RESET:
			return "VK_EVENT_RESET";
		case VK_INCOMPLETE:
			return "VK_INCOMPLETE";
		case VK_ERROR_OUT_OF_HOST_MEMORY:
			return "VK_ERROR_OUT_OF_HOST_MEMORY";
		case VK_ERROR_OUT_OF_DEVICE_MEMORY:
			return "VK_ERROR_OUT_OF_DEVICE_MEMORY";
		case VK_ERROR_INITIALIZATION_FAILED:
			return "VK_ERROR_INITIALIZATION_FAILED";
		case VK_ERROR_DEVICE_LOST:
			return "VK_ERROR_DEVICE_LOST";
		case VK_ERROR_MEMORY_MAP_FAILED:
			return "VK_ERROR_MEMORY_MAP_FAILED";
		case VK_ERROR_LAYER_NOT_PRESENT:
			return "VK_ERROR_LAYER_NOT_PRESENT";
		case VK_ERROR_EXTENSION_NOT_PRESENT:
			return "VK_ERROR_EXTENSION_NOT_PRESENT";
		case VK_ERROR_FEATURE_NOT_PRESENT:
			return "VK_ERROR_FEATURE_NOT_PRESENT";
		case VK_ERROR_INCOMPATIBLE_DRIVER:
			return "VK_ERROR_INCOMPATIBLE_DRIVER";
		case VK_ERROR_TOO_MANY_OBJECTS:
			return "VK_ERROR_TOO_MANY_OBJECTS";
		case VK_ERROR_FORMAT_NOT_SUPPORTED:
			return "VK_ERROR_FORMAT_NOT_SUPPORTED";
		case VK_ERROR_FRAGMENTED_POOL:
			return "VK_ERROR_FRAGMENTED_POOL";
		case VK_ERROR_UNKNOWN:
			return "VK_ERROR_UNKNOWN";
		case VK_ERROR_OUT_OF_POOL_MEMORY:
			return "VK_ERROR_OUT_OF_POOL_MEMORY";
		case VK_ERROR_INVALID_EXTERNAL_HANDLE:
			return "VK_ERROR_INVALID_EXTERNAL_HANDLE";
		case VK_ERROR_FRAGMENTATION:
			return "VK_ERROR_FRAGMENTATION";
		case VK_ERROR_INVALID_OPAQUE_CAPTURE_ADDRESS:
			return "VK_ERROR_INVALID_OPAQUE_CAPTURE_ADDRESS";
		case VK_ERROR_SURFACE_LOST_KHR:
			return "VK_ERROR_SURFACE_LOST_KHR";
		case VK_ERROR_NATIVE_WINDOW_IN_USE_KHR:
			return "VK_ERROR_NATIVE_WINDOW_IN_USE_KHR";
		case VK_SUBOPTIMAL_KHR:
			return "VK_SUBOPTIMAL_KHR";
		case VK_ERROR_OUT_OF_DATE_KHR:
			return "VK_ERROR_OUT_OF_DATE_KHR";
		case VK_ERROR_INCOMPATIBLE_DISPLAY_KHR:
			return "VK_ERROR_INCOMPATIBLE_DISPLAY_KHR";
		case VK_ERROR_VALIDATION_FAILED_EXT:
			return "VK_ERROR_VALIDATION_FAILED_EXT";
		case VK_ERROR_INVALID_SHADER_NV:
			return "VK_ERROR_INVALID_SHADER_NV";
		case VK_ERROR_INVALID_DRM_FORMAT_MODIFIER_PLANE_LAYOUT_EXT:
			return "VK_ERROR_INVALID_DRM_FORMAT_MODIFIER_PLANE_LAYOUT_EXT";
		case VK_ERROR_NOT_PERMITTED_EXT:
			return "VK_ERROR_NOT_PERMITTED_EXT";
		case VK_ERROR_FULL_SCREEN_EXCLUSIVE_MODE_LOST_EXT:
			return "VK_ERROR_FULL_SCREEN_EXCLUSIVE_MODE_LOST_EXT";
		case VK_THREAD_IDLE_KHR:
			return "VK_THREAD_IDLE_KHR";
		case VK_THREAD_DONE_KHR:
			return "VK_THREAD_DONE_KHR";
		case VK_OPERATION_DEFERRED_KHR:
			return "VK_OPERATION_DEFERRED_KHR";
		case VK_OPERATION_NOT_DEFERRED_KHR:
			return "VK_OPERATION_NOT_DEFERRED_KHR";
		case VK_PIPELINE_COMPILE_REQUIRED_EXT:
			return "VK_PIPELINE_COMPILE_REQUIRED_EXT";
		default:
			break;
	}
	return "Unknown result";
}

std::string std::to_string(VkFormat format)
{
	switch (format)
	{
		case VK_FORMAT_R4G4_UNORM_PACK8:
			return "VK_FORMAT_R4G4_UNORM_PACK8";
		case VK_FORMAT_R4G4B4A4_UNORM_PACK16:
			return "VK_FORMAT_R4G4B4A4_UNORM_PACK16";
		case VK_FORMAT_B4G4R4A4_UNORM_PACK16:
			return "VK_FORMAT_B4G4R4A4_UNORM_PACK16";
		case VK_FORMAT_R5G6B5_UNORM_PACK16:
			return "VK_FORMAT_R5G6B5_UNORM_PACK16";
		case VK_FORMAT_B5G6R5_UNORM_PACK16:
			return "VK_FORMAT_B5G6R5_UNORM_PACK16";
		case VK_FORMAT_R5G5B5A1_UNORM_PACK16:
			return "VK_FORMAT_R5G5B5A1_UNORM_PACK16";
		case VK_FORMAT_B5G5R5A1_UNORM_PACK16:
			return "VK_FORMAT_B5G5R5A1_UNORM_PACK16";
		case VK_FORMAT_A1R5G5B5_UNORM_PACK16:
			return "VK_FORMAT_A1R5G5B5_UNORM_PACK16";
		case VK_FORMAT_R8_UNORM:
			return "VK_FORMAT_R8_UNORM";
		case VK_FORMAT_R8_SNORM:
			return "VK_FORMAT_R8_SNORM";
		case VK_FORMAT_R8_USCALED:
			return "VK_FORMAT_R8_USCALED";
		case VK_FORMAT_R8_SSCALED:
			return "VK_FORMAT_R8_SSCALED";
		case VK_FORMAT_R8_UINT:
			return "VK_FORMAT_R8_UINT";
		case VK_FORMAT_R8_SINT:
			return "VK_FORMAT_R8_SINT";
		case VK_FORMAT_R8_SRGB:
			return "VK_FORMAT_R8_SRGB";
		case VK_FORMAT_R8G8_UNORM:
			return "VK_FORMAT_R8G8_UNORM";
		case VK_FORMAT_R8G8_SNORM:
			return "VK_FORMAT_R8G8_SNORM";
		case VK_FORMAT_R8G8_USCALED:
			return "VK_FORMAT_R8G8_USCALED";
		case VK_FORMAT_R8G8_SSCALED:
			return "VK_FORMAT_R8G8_SSCALED";
		case VK_FORMAT_R8G8_UINT:
			return "VK_FORMAT_R8G8_UINT";
		case VK_FORMAT_R8G8_SINT:
			return "VK_FORMAT_R8G8_SINT";
		case VK_FORMAT_R8G8_SRGB:
			return "VK_FORMAT_R8G8_SRGB";
		case VK_FORMAT_R8G8B8_UNORM:
			return "VK_FORMAT_R8G8B8_UNORM";
		case VK_FORMAT_R8G8B8_SNORM:
			return "VK_FORMAT_R8G8B8_SNORM";
		case VK_FORMAT_R8G8B8_USCALED:
			return "VK_FORMAT_R8G8B8_USCALED";
		case VK_FORMAT_R8G8B8_SSCALED:
			return "VK_FORMAT_R8G8B8_SSCALED";
		case VK_FORMAT_R8G8B8_UINT:
			return "VK_FORMAT_R8G8B8_UINT";
		case VK_FORMAT_R8G8B8_SINT:
			return "VK_FORMAT_R8G8B8_SINT";
		case VK_FORMAT_R8G8B8_SRGB:
			return "VK_FORMAT_R8G8B8_SRGB";
		case VK_FORMAT_B8G8R8_UNORM:
			return "VK_FORMAT_B8G8R8_UNORM";
		case VK_FORMAT_B8G8R8_SNORM:
			return "VK_FORMAT_B8G8R8_SNORM";
		case VK_FORMAT_B8G8R8_USCALED:
			return "VK_FORMAT_B8G8R8_USCALED";
		case VK_FORMAT_B8G8R8_SSCALED:
			return "VK_FORMAT_B8G8R8_SSCALED";
		case VK_FORMAT_B8G8R8_UINT:
			return "VK_FORMAT_B8G8R8_UINT";
		case VK_FORMAT_B8G8R8_SINT:
			return "VK_FORMAT_B8G8R8_SINT";
		case VK_FORMAT_B8G8R8_SRGB:
			return "VK_FORMAT_B8G8R8_SRGB";
		case VK_FORMAT_R8G8B8A8_UNORM:
			return "VK_FORMAT_R8G8B8A8_UNORM";
		case VK_FORMAT_R8G8B8A8_SNORM:
			return "VK_FORMAT_R8G8B8A8_SNORM";
		case VK_FORMAT_R8G8B8A8_USCALED:
			return "VK_FORMAT_R8G8B8A8_USCALED";
		case VK_FORMAT_R8G8B8A8_SSCALED:
			return "VK_FORMAT_R8G8B8A8_SSCALED";
		case VK_FORMAT_R8G8B8A8_UINT:
			return "VK_FORMAT_R8G8B8A8_UINT";
		case VK_FORMAT_R8G8B8A8_SINT:
			return "VK_FORMAT_R8G8B8A8_SINT";
		case VK_FORMAT_R8G8B8A8_SRGB:
			return "VK_FORMAT_R8G8B8A8_SRGB";
		case VK_FORMAT_B8G8R8A8_UNORM:
			return "VK_FORMAT_B8G8R8A8_UNORM";
		case VK_FORMAT_B8G8R8A8_SNORM:
			return "VK_FORMAT_B8G8R8A8_SNORM";
		case VK_FORMAT_B8G8R8A8_USCALED:
			return "VK_FORMAT_B8G8R8A8_USCALED";
		case VK_FORMAT_B8G8R8A8_SSCALED:
			return "VK_FORMAT_B8G8R8A8_SSCALED";
		case VK_FORMAT_B8G8R8A8_UINT:
			return "VK_FORMAT_B8G8R8A8_UINT";
		case VK_FORMAT_B8G8R8A8_SINT:
			return "VK_FORMAT_B8G8R8A8_SINT";
		case VK_FORMAT_B8G8R8A8_SRGB:
			return "VK_FORMAT_B8G8R8A8_SRGB";
		case VK_FORMAT_A8B8G8R8_UNORM_PACK32:
			return "VK_FORMAT_A8B8G8R8_UNORM_PACK32";
		case VK_FORMAT_A8B8G8R8_SNORM_PACK32:
			return "VK_FORMAT_A8B8G8R8_SNORM_PACK32";
		case VK_FORMAT_A8B8G8R8_USCALED_PACK32:
			return "VK_FORMAT_A8B8G8R8_USCALED_PACK32";
		case VK_FORMAT_A8B8G8R8_SSCALED_PACK32:
			return "VK_FORMAT_A8B8G8R8_SSCALED_PACK32";
		case VK_FORMAT_A8B8G8R8_UINT_PACK32:
			return "VK_FORMAT_A8B8G8R8_UINT_PACK32";
		case VK_FORMAT_A8B8G8R8_SINT_PACK32:
			return "VK_FORMAT_A8B8G8R8_SINT_PACK32";
		case VK_FORMAT_A8B8G8R8_SRGB_PACK32:
			return "VK_FORMAT_A8B8G8R8_SRGB_PACK32";
		case VK_FORMAT_A2R10G10B10_UNORM_PACK32:
			return "VK_FORMAT_A2R10G10B10_UNORM_PACK32";
		case VK_FORMAT_A2R10G10B10_SNORM_PACK32:
			return "VK_FORMAT_A2R10G10B10_SNORM_PACK32";
		case VK_FORMAT_A2R10G10B10_USCALED_PACK32:
			return "VK_FORMAT_A2R10G10B10_USCALED_PACK32";
		case VK_FORMAT_A2R10G10B10_SSCALED_PACK32:
			return "VK_FORMAT_A2R10G10B10_SSCALED_PACK32";
		case VK_FORMAT_A2R10G10B10_UINT_PACK32:
			return "VK_FORMAT_A2R10G10B10_UINT_PACK32";
		case VK_FORMAT_A2R10G10B10_SINT_PACK32:
			return "VK_FORMAT_A2R10G10B10_SINT_PACK32";
		case VK_FORMAT_A2B10G10R10_UNORM_PACK32:
			return "VK_FORMAT_A2B10G10R10_UNORM_PACK32";
		case VK_FORMAT_A2B10G10R10_SNORM_PACK32:
			return "VK_FORMAT_A2B10G10R10_SNORM_PACK32";
		case VK_FORMAT_A2B10G10R10_USCALED_PACK32:
			return "VK_FORMAT_A2B10G10R10_USCALED_PACK32";
		case VK_FORMAT_A2B10G10R10_SSCALED_PACK32:
			return "VK_FORMAT_A2B10G10R10_SSCALED_PACK32";
		case VK_FORMAT_A2B10G10R10_UINT_PACK32:
			return "VK_FORMAT_A2B10G10R10_UINT_PACK32";
		case VK_FORMAT_A2B10G10R10_SINT_PACK32:
			return "VK_FORMAT_A2B10G10R10_SINT_PACK32";
		case VK_FORMAT_R16_UNORM:
			return "VK_FORMAT_R16_UNORM";
		case VK_FORMAT_R16_SNORM:
			return "VK_FORMAT_R16_SNORM";
		case VK_FORMAT_R16_USCALED:
			return "VK_FORMAT_R16_USCALED";
		case VK_FORMAT_R16_SSCALED:
			return "VK_FORMAT_R16_SSCALED";
		case VK_FORMAT_R16_UINT:
			return "VK_FORMAT_R16_UINT";
		case VK_FORMAT_R16_SINT:
			return "VK_FORMAT_R16_SINT";
		case VK_FORMAT_R16_SFLOAT:
			return "VK_FORMAT_R16_SFLOAT";
		case VK_FORMAT_R16G16_UNORM:
			return "VK_FORMAT_R16G16_UNORM";
		case VK_FORMAT_R16G16_SNORM:
			return "VK_FORMAT_R16G16_SNORM";
		case VK_FORMAT_R16G16_USCALED:
			return "VK_FORMAT_R16G16_USCALED";
		case VK_FORMAT_R16G16_SSCALED:
			return "VK_FORMAT_R16G16_SSCALED";
		case VK_FORMAT_R16G16_UINT:
			return "VK_FORMAT_R16G16_UINT";
		case VK_FORMAT_R16G16_SINT:
			return "VK_FORMAT_R16G16_SINT";
		case VK_FORMAT_R16G16_SFLOAT:
			return "VK_FORMAT_R16G16_SFLOAT";
		case VK_FORMAT_R16G16B16_UNORM:
			return "VK_FORMAT_R16G16B16_UNORM";
		case VK_FORMAT_R16G16B16_SNORM:
			return "VK_FORMAT_R16G16B16_SNORM";
		case VK_FORMAT_R16G16B16_USCALED:
			return "VK_FORMAT_R16G16B16_USCALED";
		case VK_FORMAT_R16G16B16_SSCALED:
			return "VK_FORMAT_R16G16B16_SSCALED";
		case VK_FORMAT_R16G16B16_UINT:
			return "VK_FORMAT_R16G16B16_UINT";
		case VK_FORMAT_R16G16B16_SINT:
			return "VK_FORMAT_R16G16B16_SINT";
		case VK_FORMAT_R16G16B16_SFLOAT:
			return "VK_FORMAT_R16G16B16_SFLOAT";
		case VK_FORMAT_R16G16B16A16_UNORM:
			return "VK_FORMAT_R16G16B16A16_UNORM";
		case VK_FORMAT_R16G16B16A16_SNORM:
			return "VK_FORMAT_R16G16B16A16_SNORM";
		case VK_FORMAT_R16G16B16A16_USCALED:
			return "VK_FORMAT_R16G16B16A16_USCALED";
		case VK_FORMAT_R16G16B16A16_SSCALED:
			return "VK_FORMAT_R16G16B16A16_SSCALED";
		case VK_FORMAT_R16G16B16A16_UINT:
			return "VK_FORMAT_R16G16B16A16_UINT";
		case VK_FORMAT_R16G16B16A16_SINT:
			return "VK_FORMAT_R16G16B16A16_SINT";
		case VK_FORMAT_R16G16B16A16_SFLOAT:
			return "VK_FORMAT_R16G16B16A16_SFLOAT";
		case VK_FORMAT_R32_UINT:
			return "VK_FORMAT_R32_UINT";
		case VK_FORMAT_R32_SINT:
			return "VK_FORMAT_R32_SINT";
		case VK_FORMAT_R32_SFLOAT:
			return "VK_FORMAT_R32_SFLOAT";
		case VK_FORMAT_R32G32_UINT:
			return "VK_FORMAT_R32G32_UINT";
		case VK_FORMAT_R32G32_SINT:
			return "VK_FORMAT_R32G32_SINT";
		case VK_FORMAT_R32G32_SFLOAT:
			return "VK_FORMAT_R32G32_SFLOAT";
		case VK_FORMAT_R32G32B32_UINT:
			return "VK_FORMAT_R32G32B32_UINT";
		case VK_FORMAT_R32G32B32_SINT:
			return "VK_FORMAT_R32G32B32_SINT";
		case VK_FORMAT_R32G32B32_SFLOAT:
			return "VK_FORMAT_R32G32B32_SFLOAT";
		case VK_FORMAT_R32G32B32A32_UINT:
			return "VK_FORMAT_R32G32B32A32_UINT";
		case VK_FORMAT_R32G32B32A32_SINT:
			return "VK_FORMAT_R32G32B32A32_SINT";
		case VK_FORMAT_R32G32B32A32_SFLOAT:
			return "VK_FORMAT_R32G32B32A32_SFLOAT";
		case VK_FORMAT_R64_UINT:
			return "VK_FORMAT_R64_UINT";
		case VK_FORMAT_R64_SINT:
			return "VK_FORMAT_R64_SINT";
		case VK_FORMAT_R64_SFLOAT:
			return "VK_FORMAT_R64_SFLOAT";
		case VK_FORMAT_R64G64_UINT:
			return "VK_FORMAT_R64G64_UINT";
		case VK_FORMAT_R64G64_SINT:
			return "VK_FORMAT_R64G64_SINT";
		case VK_FORMAT_R64G64_SFLOAT:
			return "VK_FORMAT_R64G64_SFLOAT";
		case VK_FORMAT_R64G64B64_UINT:
			return "VK_FORMAT_R64G64B64_UINT";
		case VK_FORMAT_R64G64B64_SINT:
			return "VK_FORMAT_R64G64B64_SINT";
		case VK_FORMAT_R64G64B64_SFLOAT:
			return "VK_FORMAT_R64G64B64_SFLOAT";
		case VK_FORMAT_R64G64B64A64_UINT:
			return "VK_FORMAT_R64G64B64A64_UINT";
		case VK_FORMAT_R64G64B64A64_SINT:
			return "VK_FORMAT_R64G64B64A64_SINT";
		case VK_FORMAT_R64G64B64A64_SFLOAT:
			return "VK_FORMAT_R64G64B64A64_SFLOAT";
		case VK_FORMAT_B10G11R11_UFLOAT_PACK32:
			return "VK_FORMAT_B10G11R11_UFLOAT_PACK32";
		case VK_FORMAT_E5B9G9R9_UFLOAT_PACK32:
			return "VK_FORMAT_E5B9G9R9_UFLOAT_PACK32";
		case VK_FORMAT_D16_UNORM:
			return "VK_FORMAT_D16_UNORM";
		case VK_FORMAT_X8_D24_UNORM_PACK32:
			return "VK_FORMAT_X8_D24_UNORM_PACK32";
		case VK_FORMAT_D32_SFLOAT:
			return "VK_FORMAT_D32_SFLOAT";
		case VK_FORMAT_S8_UINT:
			return "VK_FORMAT_S8_UINT";
		case VK_FORMAT_D16_UNORM_S8_UINT:
			return "VK_FORMAT_D16_UNORM_S8_UINT";
		case VK_FORMAT_D24_UNORM_S8_UINT:
			return "VK_FORMAT_D24_UNORM_S8_UINT";
		case VK_FORMAT_D32_SFLOAT_S8_UINT:
			return "VK_FORMAT_D32_SFLOAT_S8_UINT";
		case VK_FORMAT_UNDEFINED:
			return "VK_FORMAT_UNDEFINED";
		default:
			return "VK_FORMAT_INVALID";
	}
}

std::string std::to_string(VkShaderStageFlagBits stage)
{
	switch (stage)
	{
		case VK_SHADER_STAGE_VERTEX_BIT:
			return "VERTEX";
		case VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT:
			return "TESSELLATION_CONTROL";
		case VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT:
			return "TESSELLATION_EVALUATION";
		case VK_SHADER_STAGE_GEOMETRY_BIT:
			return "GEOMETRY";
		case VK_SHADER_STAGE_FRAGMENT_BIT:
			return "FRAGMENT";
		case VK_SHADER_STAGE_COMPUTE_BIT:
			return "COMPUTE";
		case VK_SHADER_STAGE_ALL_GRAPHICS:
			return "ALL_GRAPHICS";
		case VK_SHADER_STAGE_ALL:
			return "ALL";
		case VK_SHADER_STAGE_RAYGEN_BIT_KHR:
			return "RAYGEN";
		case VK_SHADER_STAGE_ANY_HIT_BIT_KHR:
			return "ANY_HIT";
		case VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR:
			return "CLOSEST_HIT";
		case VK_SHADER_STAGE_MISS_BIT_KHR:
			return "MISS";
		case VK_SHADER_STAGE_INTERSECTION_BIT_KHR:
			return "INTERSECTION";
		case VK_SHADER_STAGE_CALLABLE_BIT_KHR:
			return "CALLABLE";
		case VK_SHADER_STAGE_TASK_BIT_NV:
			return "TASK";
		case VK_SHADER_STAGE_MESH_BIT_NV:
			return "MESH";
		default:
			break;
	}
	return "Unknown shader stage";
}