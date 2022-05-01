#pragma once

#include <volk.h>

#include <vk_mem_alloc.h>

#include <memory>
#include <string>
#include <vector>

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

class RHIDevice
{
	friend class Buffer;
	friend class Texture;
	friend class TextureView;
	friend class AccelerationStructure;
	friend class ShaderBindingTable;

  public:
	RHIDevice(Window *window);

	~RHIDevice();

	void Tick();

	std::shared_ptr<Texture>               CreateTexture(const TextureDesc &desc, const std::string &name = "");
	std::shared_ptr<Texture>               CreateTexture(const std::string &filepath);
	std::shared_ptr<TextureView>           CreateTextureView(Texture *texture, const TextureViewDesc &desc, const std::string &name = "");
	std::shared_ptr<Buffer>                CreateBuffer(const BufferDesc &desc, const std::string &name = "");
	std::shared_ptr<AccelerationStructure> CreateAccelerationStructure(const AccelerationStructureDesc &desc, const std::string &name = "");

	Texture *GetBackBuffer() const;

	VkQueue GetQueue(VkQueueFlagBits flag = VK_QUEUE_GRAPHICS_BIT) const;

  private:
	void CreateInstance();
	void CreatePhysicalDevice();
	void CreateSurface();
	void CreateLogicalDevice();
	void CreateSwapchain();

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

	VkSwapchainKHR                            m_swapchain = VK_NULL_HANDLE;
	std::vector<std::shared_ptr<Texture>>     m_swapchain_images;
	std::vector<std::shared_ptr<TextureView>> m_swapchain_image_views;

	uint32_t m_current_frame = 0;
};
}        // namespace Ilum