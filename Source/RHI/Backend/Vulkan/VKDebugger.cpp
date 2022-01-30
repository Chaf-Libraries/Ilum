#include "VKDebugger.hpp"
#include "VKInstance.hpp"

namespace Ilum::RHI::Vulkan
{
static inline VKAPI_ATTR VkBool32 VKAPI_CALL callback(VkDebugUtilsMessageSeverityFlagBitsEXT msg_severity, VkDebugUtilsMessageTypeFlagsEXT msg_type, const VkDebugUtilsMessengerCallbackDataEXT *callback_data, void *user_data)
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

void VKDebugger::Initialize(VkInstance instance)
{
	if (VKInstance::CreateDebugUtilsMessengerEXT)
	{
		VkDebugUtilsMessengerCreateInfoEXT create_info{VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT};
		create_info.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
		create_info.messageType     = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
		create_info.pfnUserCallback = callback;

		VKInstance::CreateDebugUtilsMessengerEXT(instance, &create_info, nullptr, &VKInstance::DebugUtilsMessengerEXT);
	}
}

void VKDebugger::Shutdown(VkInstance instance)
{
	if (!VKInstance::DestroyDebugUtilsMessengerEXT)
	{
		return;
	}

	VKInstance::DestroyDebugUtilsMessengerEXT(instance, VKInstance::DebugUtilsMessengerEXT, nullptr);
}

void VKDebugger::SetObjectName(uint64_t object, VkObjectType object_type, const char *name)
{
#ifndef _DEBUG
	return;
#endif        // !_DEBUG

	if (!VKInstance::SetDebugUtilsObjectNameEXT)
	{
		return;
	}

	VkDebugUtilsObjectNameInfoEXT name_info{VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT};
	name_info.pNext        = nullptr;
	name_info.objectType   = object_type;
	name_info.objectHandle = object;
	name_info.pObjectName  = name;
	//VKInstance::SetDebugUtilsObjectNameEXT(GraphicsContext::instance()->getLogicalDevice(), &name_info);
}

void VKDebugger::SetObjectTag(uint64_t object, VkObjectType object_type, uint64_t tag_name, size_t tag_size, const void *tag)
{
	if (!VKInstance::SetDebugUtilsObjectTagEXT)
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

	//VKInstance::SetDebugUtilsObjectTagEXT(GraphicsContext::instance()->getLogicalDevice(), &tag_info);
}

void VKDebugger::MarkerBegin(VkCommandBuffer cmd_buffer, const char *name, const float r, const float g, const float b, const float a)
{
	if (!VKInstance::CmdBeginDebugUtilsLabelEXT)
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
	VKInstance::CmdBeginDebugUtilsLabelEXT(cmd_buffer, &label);
}

void VKDebugger::MarkerEnd(VkCommandBuffer cmd_buffer)
{
	if (!VKInstance::CmdEndDebugUtilsLabelEXT)
	{
		return;
	}

	VKInstance::CmdEndDebugUtilsLabelEXT(cmd_buffer);
}

void VKDebugger::SetName(VkCommandPool cmd_pool, const char *name)
{
	SetObjectName((uint64_t) cmd_pool, VK_OBJECT_TYPE_COMMAND_POOL, name);
}

void VKDebugger::SetName(VkCommandBuffer cmd_buffer, const char *name)
{
	SetObjectName((uint64_t) cmd_buffer, VK_OBJECT_TYPE_COMMAND_BUFFER, name);
}

void VKDebugger::SetName(VkQueue queue, const char *name)
{
	SetObjectName((uint64_t) queue, VK_OBJECT_TYPE_QUEUE, name);
}

void VKDebugger::SetName(VkImage image, const char *name)
{
	SetObjectName((uint64_t) image, VK_OBJECT_TYPE_IMAGE, name);
}

void VKDebugger::SetName(VkImageView image_view, const char *name)
{
	SetObjectName((uint64_t) image_view, VK_OBJECT_TYPE_IMAGE_VIEW, name);
}

void VKDebugger::SetName(VkSampler sampler, const char *name)
{
	SetObjectName((uint64_t) sampler, VK_OBJECT_TYPE_SAMPLER, name);
}

void VKDebugger::SetName(VkBuffer buffer, const char *name)
{
	SetObjectName((uint64_t) buffer, VK_OBJECT_TYPE_BUFFER, name);
}

void VKDebugger::SetName(VkBufferView buffer_view, const char *name)
{
	SetObjectName((uint64_t) buffer_view, VK_OBJECT_TYPE_BUFFER_VIEW, name);
}

void VKDebugger::SetName(VkDeviceMemory memory, const char *name)
{
	SetObjectName((uint64_t) memory, VK_OBJECT_TYPE_DEVICE_MEMORY, name);
}

void VKDebugger::SetName(VkAccelerationStructureKHR acceleration_structure, const char *name)
{
	SetObjectName((uint64_t) acceleration_structure, VK_OBJECT_TYPE_ACCELERATION_STRUCTURE_KHR, name);
}

void VKDebugger::SetName(VkShaderModule shader_module, const char *name)
{
	SetObjectName((uint64_t) shader_module, VK_OBJECT_TYPE_SHADER_MODULE, name);
}

void VKDebugger::SetName(VkPipeline pipeline, const char *name)
{
	SetObjectName((uint64_t) pipeline, VK_OBJECT_TYPE_PIPELINE, name);
}

void VKDebugger::SetName(VkPipelineLayout pipeline_layout, const char *name)
{
	SetObjectName((uint64_t) pipeline_layout, VK_OBJECT_TYPE_PIPELINE_LAYOUT, name);
}

void VKDebugger::SetName(VkRenderPass render_pass, const char *name)
{
	SetObjectName((uint64_t) render_pass, VK_OBJECT_TYPE_RENDER_PASS, name);
}

void VKDebugger::SetName(VkFramebuffer frame_buffer, const char *name)
{
	SetObjectName((uint64_t) frame_buffer, VK_OBJECT_TYPE_FRAMEBUFFER, name);
}

void VKDebugger::SetName(VkDescriptorSetLayout descriptor_set_layout, const char *name)
{
	SetObjectName((uint64_t) descriptor_set_layout, VK_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT, name);
}

void VKDebugger::SetName(VkDescriptorSet descriptor_set, const char *name)
{
	SetObjectName((uint64_t) descriptor_set, VK_OBJECT_TYPE_DESCRIPTOR_SET, name);
}

void VKDebugger::SetName(VkDescriptorPool descriptor_pool, const char *name)
{
	SetObjectName((uint64_t) descriptor_pool, VK_OBJECT_TYPE_DESCRIPTOR_POOL, name);
}

void VKDebugger::SetName(VkSemaphore semaphore, const char *name)
{
	SetObjectName((uint64_t) semaphore, VK_OBJECT_TYPE_SEMAPHORE, name);
}

void VKDebugger::SetName(VkFence fence, const char *name)
{
	SetObjectName((uint64_t) fence, VK_OBJECT_TYPE_FENCE, name);
}

void VKDebugger::SetName(VkEvent event, const char *name)
{
	SetObjectName((uint64_t) event, VK_OBJECT_TYPE_EVENT, name);
}
}