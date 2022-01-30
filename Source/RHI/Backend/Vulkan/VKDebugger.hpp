#pragma once

#include "Vulkan.hpp"

namespace Ilum::RHI::Vulkan
{
class VKDebugger
{
  public:
	VKDebugger()  = default;
	~VKDebugger() = default;

	static void Initialize(VkInstance instance);
	static void Shutdown(VkInstance instance);
	static void SetObjectName(uint64_t object, VkObjectType object_type, const char *name);
	static void SetObjectTag(uint64_t object, VkObjectType object_type, uint64_t tag_name, size_t tag_size, const void *tag);
	static void MarkerBegin(VkCommandBuffer cmd_buffer, const char *name, const float r, const float g, const float b, const float a);
	static void MarkerEnd(VkCommandBuffer cmd_buffer);

	static void SetName(VkCommandPool cmd_pool, const char *name);
	static void SetName(VkCommandBuffer cmd_buffer, const char *name);
	static void SetName(VkQueue queue, const char *name);
	static void SetName(VkImage image, const char *name);
	static void SetName(VkImageView image_view, const char *name);
	static void SetName(VkSampler sampler, const char *name);
	static void SetName(VkBuffer buffer, const char *name);
	static void SetName(VkBufferView buffer_view, const char *name);
	static void SetName(VkDeviceMemory memory, const char *name);
	static void SetName(VkAccelerationStructureKHR acceleration_structure, const char *name);
	static void SetName(VkShaderModule shader_module, const char *name);
	static void SetName(VkPipeline pipeline, const char *name);
	static void SetName(VkPipelineLayout pipeline_layout, const char *name);
	static void SetName(VkRenderPass render_pass, const char *name);
	static void SetName(VkFramebuffer frame_buffer, const char *name);
	static void SetName(VkDescriptorSetLayout descriptor_set_layout, const char *name);
	static void SetName(VkDescriptorSet descriptor_set, const char *name);
	static void SetName(VkDescriptorPool descriptor_pool, const char *name);
	static void SetName(VkSemaphore semaphore, const char *name);
	static void SetName(VkFence fence, const char *name);
	static void SetName(VkEvent event, const char *name);
};
}