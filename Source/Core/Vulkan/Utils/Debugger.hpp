#pragma once

#include <volk.h>

#include "Math/Vector4.h"

namespace Ilum::Vulkan
{
class Debugger
{
  public:
	Debugger()  = default;
	~Debugger() = default;

	static void initialize(VkInstance instance);
	static void shutdown(VkInstance instance);
	static void setObjectName(VkDevice device, uint64_t object, VkObjectType object_type, const char *name);
	static void setObjectTag(VkDevice device, uint64_t object, VkObjectType object_type, uint64_t tag_name, size_t tag_size, const void *tag);
	static void markerBegin(VkDevice device, VkCommandBuffer cmd_buffer, const char *name, const Math::Rgba &color);
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
}        // namespace Ilum::Vulkan