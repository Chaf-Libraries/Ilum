#pragma once

#include <volk.h>

#define VK_CHECK(result) Ilum::Vulkan::check(result)
#define VK_ASSERT(result) Ilum::Vulkan::_assert(result)

namespace Ilum::Vulkan
{
const bool check(VkResult result);
void _assert(VkResult result);

class FunctionEXT
{
  public:
	FunctionEXT() = default;

	~FunctionEXT() = default;

	static void initialzize(VkInstance instance);

	static PFN_vkCreateDebugUtilsMessengerEXT          create_debug_utils_messenger;
	static VkDebugUtilsMessengerEXT                    debug_utils_messenger;
	static PFN_vkDestroyDebugUtilsMessengerEXT         destroy_messenger;
	static PFN_vkSetDebugUtilsObjectTagEXT             set_debug_utils_object_tag;
	static PFN_vkSetDebugUtilsObjectNameEXT            set_debug_utils_object_name;
	static PFN_vkCmdBeginDebugUtilsLabelEXT            begin_debug_utils_label;
	static PFN_vkCmdEndDebugUtilsLabelEXT              end_debug_utils_label;
	static PFN_vkGetPhysicalDeviceMemoryProperties2KHR get_physical_device_memory_properties_2;
};

class Debugger
{
  public:
	Debugger()  = default;
	~Debugger() = default;

	static void initialize(VkInstance instance);
	static void shutdown(VkInstance instance);
	static void setObjectName(VkDevice device, uint64_t object, VkObjectType object_type, const char *name);
	static void setObjectTag(VkDevice device, uint64_t object, VkObjectType object_type, uint64_t tag_name, size_t tag_size, const void *tag);
	static void markerBegin(VkDevice device, VkCommandBuffer cmd_buffer, const char *name, const float r, const float g, const float b, const float a);
	static void markerEnd(VkDevice device, VkCommandBuffer cmd_buffer);

	static void setName(VkDevice device, VkCommandPool cmd_pool, const char *name);
	static void setName(VkDevice device, VkCommandBuffer cmd_buffer, const char *name);
	static void setName(VkDevice device, VkQueue queue, const char *name);
	static void setName(VkDevice device, VkImage image, const char *name);
	static void setName(VkDevice device, VkImageView image_view, const char *name);
	static void setName(VkDevice device, VkSampler sampler, const char *name);
	static void setName(VkDevice device, VkBuffer buffer, const char *name);
	static void setName(VkDevice device, VkBufferView buffer_view, const char *name);
	static void setName(VkDevice device, VkDeviceMemory memory, const char *name);
	static void setName(VkDevice device, VkAccelerationStructureKHR acceleration_structure, const char *name);
	static void setName(VkDevice device, VkShaderModule shader_module, const char *name);
	static void setName(VkDevice device, VkPipeline pipeline, const char *name);
	static void setName(VkDevice device, VkPipelineLayout pipeline_layout, const char *name);
	static void setName(VkDevice device, VkRenderPass render_pass, const char *name);
	static void setName(VkDevice device, VkFramebuffer frame_buffer, const char *name);
	static void setName(VkDevice device, VkDescriptorSetLayout descriptor_set_layout, const char *name);
	static void setName(VkDevice device, VkDescriptorSet descriptor_set, const char *name);
	static void setName(VkDevice device, VkDescriptorPool descriptor_pool, const char *name);
	static void setName(VkDevice device, VkSemaphore semaphore, const char *name);
	static void setName(VkDevice device, VkFence fence, const char *name);
	static void setName(VkDevice device, VkEvent event, const char *name);
};
}