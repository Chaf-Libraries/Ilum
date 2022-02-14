#pragma once

#include <volk.h>
#include <vk_mem_alloc.h>

#include <string>
#include <vector>

#include <Core/Logger/Logger.hpp>

namespace Ilum::Graphics
{
void vk_assert(VkResult result);

const bool vk_check(VkResult result);

std::string shader_stage_to_string(VkShaderStageFlags stage);

bool IsDepth(VkFormat format);
bool IsStencil(VkFormat format);
bool IsDepthStencil(VkFormat format);

struct InstanceExtension
{
	static const std::vector<const char *>                 extensions;
	static const std::vector<const char *>                 validation_layers;
	static const std::vector<VkValidationFeatureEnableEXT> validation_extensions;
};

struct InstanceFunctionEXT
{
	static PFN_vkCreateDebugUtilsMessengerEXT          CreateDebugUtilsMessengerEXT;
	static VkDebugUtilsMessengerEXT                    DebugUtilsMessengerEXT;
	static PFN_vkDestroyDebugUtilsMessengerEXT         DestroyDebugUtilsMessengerEXT;
	static PFN_vkSetDebugUtilsObjectTagEXT             SetDebugUtilsObjectTagEXT;
	static PFN_vkSetDebugUtilsObjectNameEXT            SetDebugUtilsObjectNameEXT;
	static PFN_vkCmdBeginDebugUtilsLabelEXT            CmdBeginDebugUtilsLabelEXT;
	static PFN_vkCmdEndDebugUtilsLabelEXT              CmdEndDebugUtilsLabelEXT;
	static PFN_vkGetPhysicalDeviceMemoryProperties2KHR GetPhysicalDeviceMemoryProperties2KHR;
};

struct DeviceExtension
{
	static const std::vector<const char *> extensions;
};

enum class QueueFamily
{
	Graphics,
	Compute,
	Transfer,
	Present
};

class VKDebugger
{
  public:
	VKDebugger()  = default;
	~VKDebugger() = default;

	static void Initialize(VkInstance instance);
	static void Shutdown(VkInstance instance);
	static void SetObjectName(VkDevice device, uint64_t object, VkObjectType object_type, const char *name);
	static void SetObjectTag(VkDevice device, uint64_t object, VkObjectType object_type, uint64_t tag_name, size_t tag_size, const void *tag);
	static void MarkerBegin(VkCommandBuffer cmd_buffer, const char *name, const float r, const float g, const float b, const float a);
	static void MarkerEnd(VkCommandBuffer cmd_buffer);

	static void SetName(VkDevice device, VkCommandPool cmd_pool, const char *name);
	static void SetName(VkDevice device, VkCommandBuffer cmd_buffer, const char *name);
	static void SetName(VkDevice device, VkQueue queue, const char *name);
	static void SetName(VkDevice device, VkImage image, const char *name);
	static void SetName(VkDevice device, VkImageView image_view, const char *name);
	static void SetName(VkDevice device, VkSampler sampler, const char *name);
	static void SetName(VkDevice device, VkBuffer buffer, const char *name);
	static void SetName(VkDevice device, VkBufferView buffer_view, const char *name);
	static void SetName(VkDevice device, VkDeviceMemory memory, const char *name);
	static void SetName(VkDevice device, VkAccelerationStructureKHR acceleration_structure, const char *name);
	static void SetName(VkDevice device, VkShaderModule shader_module, const char *name);
	static void SetName(VkDevice device, VkPipeline pipeline, const char *name);
	static void SetName(VkDevice device, VkPipelineLayout pipeline_layout, const char *name);
	static void SetName(VkDevice device, VkRenderPass render_pass, const char *name);
	static void SetName(VkDevice device, VkFramebuffer frame_buffer, const char *name);
	static void SetName(VkDevice device, VkDescriptorSetLayout descriptor_set_layout, const char *name);
	static void SetName(VkDevice device, VkDescriptorSet descriptor_set, const char *name);
	static void SetName(VkDevice device, VkDescriptorPool descriptor_pool, const char *name);
	static void SetName(VkDevice device, VkSemaphore semaphore, const char *name);
	static void SetName(VkDevice device, VkFence fence, const char *name);
	static void SetName(VkDevice device, VkEvent event, const char *name);
};
}        // namespace Ilum::Graphics

namespace std
{
std::string to_string(VkResult result);

std::string to_string(VkFormat format);

std::string to_string(VkShaderStageFlagBits stage);
}        // namespace std

#define VK_CHECK(result) Ilum::Graphics::vk_check(result)
#define VK_ASSERT(result) Ilum::Graphics::vk_assert(result)