#include "Debugger.hpp"
#include "FunctionEXT.hpp"
#include "Logger.hpp"

namespace Ilum::Vulkan
{
static inline VKAPI_ATTR VkBool32 VKAPI_CALL callback(VkDebugUtilsMessageSeverityFlagBitsEXT msg_severity, VkDebugUtilsMessageTypeFlagsEXT msg_type, const VkDebugUtilsMessengerCallbackDataEXT *callback_data, void *user_data)
{
	if (msg_severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT)
	{
		VK_INFO(callback_data->pMessage);
	}
	else if (msg_severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT)
	{
		VK_WARN(callback_data->pMessage);
	}
	else if (msg_severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT)
	{
		VK_ERROR(callback_data->pMessage);
	}

	return VK_FALSE;
}

void Debugger::initialize(VkInstance instance)
{
	if (FunctionEXT::create_debug_utils_messenger)
	{
		VkDebugUtilsMessengerCreateInfoEXT create_info{VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT};
		create_info.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
		create_info.messageType     = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
		create_info.pfnUserCallback = callback;

		FunctionEXT::create_debug_utils_messenger(instance, &create_info, nullptr, &FunctionEXT::debug_utils_messenger);
	}
}

void Debugger::shutdown(VkInstance instance)
{
	if (!FunctionEXT::destroy_messenger)
	{
		return;
	}

	FunctionEXT::destroy_messenger(instance, FunctionEXT::debug_utils_messenger, nullptr);
}

void Debugger::setObjectName(VkDevice device, uint64_t object, VkObjectType object_type, const char *name)
{
#ifndef _DEBUG
	return;
#endif        // !_DEBUG

	if (!FunctionEXT::set_debug_utils_object_name)
	{
		return;
	}

	VkDebugUtilsObjectNameInfoEXT name_info{VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT};
	name_info.pNext        = nullptr;
	name_info.objectType   = object_type;
	name_info.objectHandle = object;
	name_info.pObjectName  = name;
	FunctionEXT::set_debug_utils_object_name(device, &name_info);
}

void Debugger::setObjectTag(VkDevice device, uint64_t object, VkObjectType object_type, uint64_t tag_name, size_t tag_size, const void *tag)
{
	if (!FunctionEXT::set_debug_utils_object_tag)
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

	FunctionEXT::set_debug_utils_object_tag(device, &tag_info);
}

void Debugger::markerBegin(VkDevice device, VkCommandBuffer cmd_buffer, const char *name, const Math::Rgba &color)
{
	if (!FunctionEXT::begin_debug_utils_label)
	{
		return;
	}

	VkDebugUtilsLabelEXT label{VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT};
	label.pNext      = nullptr;
	label.pLabelName = name;
	label.color[0]   = color.x;
	label.color[1]   = color.y;
	label.color[2]   = color.z;
	label.color[3]   = color.w;
	FunctionEXT::begin_debug_utils_label(cmd_buffer, &label);
}

void Debugger::markerEnd(VkDevice device, VkCommandBuffer cmd_buffer)
{
	if (!FunctionEXT::end_debug_utils_label)
	{
		return;
	}

	FunctionEXT::end_debug_utils_label(cmd_buffer);
}

void Debugger::setName(VkDevice device, VkCommandPool cmd_pool, const char *name)
{
	setObjectName(device, (uint64_t) cmd_pool, VK_OBJECT_TYPE_COMMAND_POOL, name);
}

void Debugger::setName(VkDevice device, VkCommandBuffer cmd_buffer, const char *name)
{
	setObjectName(device, (uint64_t) cmd_buffer, VK_OBJECT_TYPE_COMMAND_BUFFER, name);
}

void Debugger::setName(VkDevice device, VkQueue queue, const char *name)
{
	setObjectName(device, (uint64_t) queue, VK_OBJECT_TYPE_QUEUE, name);
}

void Debugger::setName(VkDevice device, VkImage image, const char *name)
{
	setObjectName(device, (uint64_t) image, VK_OBJECT_TYPE_IMAGE, name);
}

void Debugger::setName(VkDevice device, VkImageView image_view, const char *name)
{
	setObjectName(device, (uint64_t) image_view, VK_OBJECT_TYPE_IMAGE_VIEW, name);
}

void Debugger::setName(VkDevice device, VkSampler sampler, const char *name)
{
	setObjectName(device, (uint64_t) sampler, VK_OBJECT_TYPE_SAMPLER, name);
}

void Debugger::setName(VkDevice device, VkBuffer buffer, const char *name)
{
	setObjectName(device, (uint64_t) buffer, VK_OBJECT_TYPE_BUFFER, name);
}

void Debugger::setName(VkDevice device, VkBufferView buffer_view, const char *name)
{
	setObjectName(device, (uint64_t) buffer_view, VK_OBJECT_TYPE_BUFFER_VIEW, name);
}

void Debugger::setName(VkDevice device, VkDeviceMemory memory, const char *name)
{
	setObjectName(device, (uint64_t) memory, VK_OBJECT_TYPE_DEVICE_MEMORY, name);
}

void Debugger::setName(VkDevice device, VkAccelerationStructureKHR acceleration_structure, const char *name)
{
	setObjectName(device, (uint64_t) acceleration_structure, VK_OBJECT_TYPE_ACCELERATION_STRUCTURE_KHR, name);
}

void Debugger::setName(VkDevice device, VkShaderModule shader_module, const char *name)
{
	setObjectName(device, (uint64_t) shader_module, VK_OBJECT_TYPE_SHADER_MODULE, name);
}

void Debugger::setName(VkDevice device, VkPipeline pipeline, const char *name)
{
	setObjectName(device, (uint64_t) pipeline, VK_OBJECT_TYPE_PIPELINE, name);
}

void Debugger::setName(VkDevice device, VkPipelineLayout pipeline_layout, const char *name)
{
	setObjectName(device, (uint64_t) pipeline_layout, VK_OBJECT_TYPE_PIPELINE_LAYOUT, name);
}

void Debugger::setName(VkDevice device, VkRenderPass render_pass, const char *name)
{
	setObjectName(device, (uint64_t) render_pass, VK_OBJECT_TYPE_RENDER_PASS, name);
}

void Debugger::setName(VkDevice device, VkFramebuffer frame_buffer, const char *name)
{
	setObjectName(device, (uint64_t) frame_buffer, VK_OBJECT_TYPE_FRAMEBUFFER, name);
}

void Debugger::setName(VkDevice device, VkDescriptorSetLayout descriptor_set_layout, const char *name)
{
	setObjectName(device, (uint64_t) descriptor_set_layout, VK_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT, name);
}

void Debugger::setName(VkDevice device, VkDescriptorSet descriptor_set, const char *name)
{
	setObjectName(device, (uint64_t) descriptor_set, VK_OBJECT_TYPE_DESCRIPTOR_SET, name);
}

void Debugger::setName(VkDevice device, VkDescriptorPool descriptor_pool, const char *name)
{
	setObjectName(device, (uint64_t) descriptor_pool, VK_OBJECT_TYPE_DESCRIPTOR_POOL, name);
}

void Debugger::setName(VkDevice device, VkSemaphore semaphore, const char *name)
{
	setObjectName(device, (uint64_t) semaphore, VK_OBJECT_TYPE_SEMAPHORE, name);
}

void Debugger::setName(VkDevice device, VkFence fence, const char *name)
{
	setObjectName(device, (uint64_t) fence, VK_OBJECT_TYPE_FENCE, name);
}

void Debugger::setName(VkDevice device, VkEvent event, const char *name)
{
	setObjectName(device, (uint64_t) event, VK_OBJECT_TYPE_EVENT, name);
}
}        // namespace Ilum::Vulkan