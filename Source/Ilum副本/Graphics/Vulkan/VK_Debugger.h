#pragma once

#include "Vulkan.hpp"

namespace Ilum
{
class VK_Debugger
{
  public:
	VK_Debugger() = default;
	~VK_Debugger() = default;

	static void initialize(VkInstance instance);
	static void shutdown(VkInstance instance);
	static void setObjectName(uint64_t object, VkObjectType object_type, const char *name);
	static void setObjectTag(uint64_t object, VkObjectType object_type, uint64_t tag_name, size_t tag_size, const void *tag);
	static void markerBegin(VkCommandBuffer cmd_buffer, const char *name, const float r, const float g, const float b, const float a);
	static void markerEnd(VkCommandBuffer cmd_buffer);

	static void setName(VkCommandPool cmd_pool, const char *name);
	static void setName(VkCommandBuffer cmd_buffer, const char *name);
	static void setName(VkQueue queue, const char *name);
	static void setName(VkImage image, const char *name);
	static void setName(VkImageView image_view, const char *name);
	static void setName(VkSampler sampler, const char *name);
	static void setName(VkBuffer buffer, const char *name);
	static void setName(VkBufferView buffer_view, const char *name);
	static void setName(VkDeviceMemory memory, const char *name);
	static void setName(VkAccelerationStructureKHR acceleration_structure, const char *name);
	static void setName(VkShaderModule shader_module, const char *name);
	static void setName(VkPipeline pipeline, const char *name);
	static void setName(VkPipelineLayout pipeline_layout, const char *name);
	static void setName(VkRenderPass render_pass, const char *name);
	static void setName(VkFramebuffer frame_buffer, const char *name);
	static void setName(VkDescriptorSetLayout descriptor_set_layout, const char *name);
	static void setName(VkDescriptorSet descriptor_set, const char *name);
	static void setName(VkDescriptorPool descriptor_pool, const char *name);
	static void setName(VkSemaphore semaphore, const char *name);
	static void setName(VkFence fence, const char *name);
	static void setName(VkEvent event, const char *name);
};
}