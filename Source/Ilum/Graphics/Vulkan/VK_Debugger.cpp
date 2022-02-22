#include "VK_Debugger.h"

#include "Device/Instance.hpp"
#include "Device/LogicalDevice.hpp"

#include "Graphics/GraphicsContext.hpp"

namespace Ilum
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

void VK_Debugger::initialize(VkInstance instance)
{
	if (Instance::createDebugUtilsMessengerEXT)
	{
		VkDebugUtilsMessengerCreateInfoEXT create_info{VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT};
		create_info.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
		create_info.messageType     = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
		create_info.pfnUserCallback = callback;

		Instance::createDebugUtilsMessengerEXT(instance, &create_info, nullptr, &Instance::debugUtilsMessengerEXT);
	}
}

void VK_Debugger::shutdown(VkInstance instance)
{
	if (!Instance::destroyDebugUtilsMessengerEXT)
	{
		return;
	}

	Instance::destroyDebugUtilsMessengerEXT(instance, Instance::debugUtilsMessengerEXT, nullptr);
}

void VK_Debugger::setObjectName(uint64_t object, VkObjectType object_type, const char *name)
{
#ifndef _DEBUG
	return;
#endif        // !_DEBUG

	if (!Instance::setDebugUtilsObjectNameEXT)
	{
		return;
	}

	if (object == (uint64_t)VK_NULL_HANDLE)
	{
		return;
	}

	VkDebugUtilsObjectNameInfoEXT name_info{VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT};
	name_info.pNext        = nullptr;
	name_info.objectType   = object_type;
	name_info.objectHandle = object;
	name_info.pObjectName  = name;
	Instance::setDebugUtilsObjectNameEXT(GraphicsContext::instance()->getLogicalDevice(), &name_info);
}

void VK_Debugger::setObjectTag(uint64_t object, VkObjectType object_type, uint64_t tag_name, size_t tag_size, const void *tag)
{
	if (!Instance::setDebugUtilsObjectTagEXT)
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

	Instance::setDebugUtilsObjectTagEXT(GraphicsContext::instance()->getLogicalDevice(), &tag_info);
}

void VK_Debugger::markerBegin(VkCommandBuffer cmd_buffer, const char *name, const float r, const float g, const float b, const float a)
{
	if (!Instance::cmdBeginDebugUtilsLabelEXT)
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
	Instance::cmdBeginDebugUtilsLabelEXT(cmd_buffer, &label);
}

void VK_Debugger::markerEnd(VkCommandBuffer cmd_buffer)
{
	if (!Instance::cmdEndDebugUtilsLabelEXT)
	{
		return;
	}

	Instance::cmdEndDebugUtilsLabelEXT(cmd_buffer);
}

void VK_Debugger::setName(VkCommandPool cmd_pool, const char *name)
{
	setObjectName((uint64_t) cmd_pool, VK_OBJECT_TYPE_COMMAND_POOL, name);
}

void VK_Debugger::setName(VkCommandBuffer cmd_buffer, const char *name)
{
	setObjectName((uint64_t) cmd_buffer, VK_OBJECT_TYPE_COMMAND_BUFFER, name);
}

void VK_Debugger::setName(VkQueue queue, const char *name)
{
	setObjectName((uint64_t) queue, VK_OBJECT_TYPE_QUEUE, name);
}

void VK_Debugger::setName(VkImage image, const char *name)
{
	setObjectName((uint64_t) image, VK_OBJECT_TYPE_IMAGE, name);
}

void VK_Debugger::setName(VkImageView image_view, const char *name)
{
	setObjectName((uint64_t) image_view, VK_OBJECT_TYPE_IMAGE_VIEW, name);
}

void VK_Debugger::setName(VkSampler sampler, const char *name)
{
	setObjectName((uint64_t) sampler, VK_OBJECT_TYPE_SAMPLER, name);
}

void VK_Debugger::setName(VkBuffer buffer, const char *name)
{
	setObjectName((uint64_t) buffer, VK_OBJECT_TYPE_BUFFER, name);
}

void VK_Debugger::setName(VkBufferView buffer_view, const char *name)
{
	setObjectName((uint64_t) buffer_view, VK_OBJECT_TYPE_BUFFER_VIEW, name);
}

void VK_Debugger::setName(VkDeviceMemory memory, const char *name)
{
	setObjectName((uint64_t) memory, VK_OBJECT_TYPE_DEVICE_MEMORY, name);
}

void VK_Debugger::setName(VkAccelerationStructureKHR acceleration_structure, const char *name)
{
	setObjectName((uint64_t) acceleration_structure, VK_OBJECT_TYPE_ACCELERATION_STRUCTURE_KHR, name);
}

void VK_Debugger::setName(VkShaderModule shader_module, const char *name)
{
	setObjectName((uint64_t) shader_module, VK_OBJECT_TYPE_SHADER_MODULE, name);
}

void VK_Debugger::setName(VkPipeline pipeline, const char *name)
{
	setObjectName((uint64_t) pipeline, VK_OBJECT_TYPE_PIPELINE, name);
}

void VK_Debugger::setName(VkPipelineLayout pipeline_layout, const char *name)
{
	setObjectName((uint64_t) pipeline_layout, VK_OBJECT_TYPE_PIPELINE_LAYOUT, name);
}

void VK_Debugger::setName(VkRenderPass render_pass, const char *name)
{
	setObjectName((uint64_t) render_pass, VK_OBJECT_TYPE_RENDER_PASS, name);
}

void VK_Debugger::setName(VkFramebuffer frame_buffer, const char *name)
{
	setObjectName((uint64_t) frame_buffer, VK_OBJECT_TYPE_FRAMEBUFFER, name);
}

void VK_Debugger::setName(VkDescriptorSetLayout descriptor_set_layout, const char *name)
{
	setObjectName((uint64_t) descriptor_set_layout, VK_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT, name);
}

void VK_Debugger::setName(VkDescriptorSet descriptor_set, const char *name)
{
	setObjectName((uint64_t) descriptor_set, VK_OBJECT_TYPE_DESCRIPTOR_SET, name);
}

void VK_Debugger::setName(VkDescriptorPool descriptor_pool, const char *name)
{
	setObjectName((uint64_t) descriptor_pool, VK_OBJECT_TYPE_DESCRIPTOR_POOL, name);
}

void VK_Debugger::setName(VkSemaphore semaphore, const char *name)
{
	setObjectName((uint64_t) semaphore, VK_OBJECT_TYPE_SEMAPHORE, name);
}

void VK_Debugger::setName(VkFence fence, const char *name)
{
	setObjectName((uint64_t) fence, VK_OBJECT_TYPE_FENCE, name);
}

void VK_Debugger::setName(VkEvent event, const char *name)
{
	setObjectName((uint64_t) event, VK_OBJECT_TYPE_EVENT, name);
}
}        // namespace Ilum