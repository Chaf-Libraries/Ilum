#pragma once

#include <volk.h>

#include <vk_mem_alloc.h>

#include "ShaderCompiler.hpp"
#include "ShaderReflection.hpp"

namespace Ilum
{
class Window;
struct TextureDesc;
class Texture;
struct TextureViewDesc;
class TextureView;
struct BufferDesc;
class Buffer;
struct AccelerationStructureDesc;
class AccelerationStructure;
class ShaderBindingTable;
class DescriptorSetLayout;
enum class ShaderType;
class ShaderAllocator;
class DescriptorAllocator;
class Frame;

class RHIDevice
{
	friend class Buffer;
	friend class Texture;
	friend class TextureView;
	friend class AccelerationStructure;
	friend class ShaderBindingTable;
	friend class DescriptorLayout;
	friend class DescriptorPool;
	friend class ShaderAllocator;
	friend class CommandPool;
	friend class CommandBuffer;
	friend class FenceAllocator;
	friend class SemaphoreAllocator;
	friend class Frame;
	friend class Sampler;
	friend class ImGuiContext;

  public:
	RHIDevice(Window *window);

	~RHIDevice();

	Texture *GetBackBuffer() const;

	VkQueue GetQueue(VkQueueFlagBits flag = VK_QUEUE_GRAPHICS_BIT) const;

	VkFormat GetSwapchainFormat() const;
	VkFormat GetDepthStencilFormat() const;

	VkShaderModule       LoadShader(const ShaderDesc &desc);
	ShaderReflectionData ReflectShader(VkShaderModule shader);

	CommandBuffer &RequestCommandBuffer(VkCommandBufferLevel level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, VkQueueFlagBits queue = VK_QUEUE_GRAPHICS_BIT);

	uint32_t GetGraphicsFamily() const;
	uint32_t GetComputeFamily() const;
	uint32_t GetTransferFamily() const;
	uint32_t GetPresentFamily() const;

	uint32_t GetCurrentFrame() const;

	void NewFrame();
	void Submit(CommandBuffer& cmd_buffer);
	void SubmitIdle(CommandBuffer &cmd_buffer, VkQueueFlagBits queue = VK_QUEUE_GRAPHICS_BIT);
	void EndFrame();

  private:
	void CreateInstance();
	void CreatePhysicalDevice();
	void CreateSurface();
	void CreateLogicalDevice();
	void CreateSwapchain();
	void CreateAllocator();

  private:
	// Extension functions
	static PFN_vkCreateDebugUtilsMessengerEXT          vkCreateDebugUtilsMessengerEXT;
	static VkDebugUtilsMessengerEXT                    vkDebugUtilsMessengerEXT;
	static PFN_vkDestroyDebugUtilsMessengerEXT         vkDestroyDebugUtilsMessengerEXT;
	static PFN_vkSetDebugUtilsObjectTagEXT             vkSetDebugUtilsObjectTagEXT;
	static PFN_vkSetDebugUtilsObjectNameEXT            vkSetDebugUtilsObjectNameEXT;
	static PFN_vkCmdBeginDebugUtilsLabelEXT            vkCmdBeginDebugUtilsLabelEXT;
	static PFN_vkCmdEndDebugUtilsLabelEXT              vkCmdEndDebugUtilsLabelEXT;
	static PFN_vkGetPhysicalDeviceMemoryProperties2KHR vkGetPhysicalDeviceMemoryProperties2KHR;

  private:
	static const std::vector<const char *>                 s_instance_extensions;
	static const std::vector<const char *>                 s_validation_layers;
	static const std::vector<VkValidationFeatureEnableEXT> s_validation_extensions;
	static const std::vector<const char *>                 s_device_extensions;

  private:
	Window *p_window;

	VkInstance       m_instance        = VK_NULL_HANDLE;
	VkPhysicalDevice m_physical_device = VK_NULL_HANDLE;
	VkSurfaceKHR     m_surface         = VK_NULL_HANDLE;
	VkDevice         m_device          = VK_NULL_HANDLE;
	VmaAllocator     m_allocator       = VK_NULL_HANDLE;

	VkQueue m_graphics_queue = VK_NULL_HANDLE;
	VkQueue m_compute_queue  = VK_NULL_HANDLE;
	VkQueue m_transfer_queue = VK_NULL_HANDLE;
	VkQueue m_present_queue  = VK_NULL_HANDLE;

	uint32_t m_graphics_family;
	uint32_t m_compute_family;
	uint32_t m_transfer_family;
	uint32_t m_present_family;

	VkSwapchainKHR                        m_swapchain = VK_NULL_HANDLE;
	std::vector<std::shared_ptr<Texture>> m_swapchain_images;

	VkFormat m_swapchain_format = VK_FORMAT_UNDEFINED;
	VkFormat m_depth_format = VK_FORMAT_UNDEFINED;

	std::unique_ptr<ShaderAllocator>     m_shader_allocator     = nullptr;
	std::unique_ptr<DescriptorAllocator> m_descriptor_allocator = nullptr;

	std::vector<std::unique_ptr<Frame>> m_frames;

	std::vector<VkSemaphore> m_present_complete;
	std::vector<VkSemaphore> m_render_complete;

	std::vector<VkCommandBuffer> m_cmd_buffer_for_submit;

	VkPipelineCache m_pipeline_cache = VK_NULL_HANDLE;

	uint32_t m_current_frame = 0;
};
}        // namespace Ilum