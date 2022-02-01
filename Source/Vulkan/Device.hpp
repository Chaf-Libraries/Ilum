#pragma once

#include "Vulkan.hpp"

#include <map>

namespace Ilum::Vulkan
{
class CommandPool;

// Vulkan Instance
class Instance
{
  public:
	Instance();
	~Instance();

	Instance(const Instance &) = delete;
	Instance &operator=(const Instance &) = delete;
	Instance(Instance &&)                 = delete;
	Instance &operator=(Instance &&) = delete;

	operator const VkInstance &() const;

	const VkInstance &GetHandle() const;

  public:
	// Extension functions
	static PFN_vkCreateDebugUtilsMessengerEXT          CreateDebugUtilsMessengerEXT;
	static VkDebugUtilsMessengerEXT                    DebugUtilsMessengerEXT;
	static PFN_vkDestroyDebugUtilsMessengerEXT         DestroyDebugUtilsMessengerEXT;
	static PFN_vkSetDebugUtilsObjectTagEXT             SetDebugUtilsObjectTagEXT;
	static PFN_vkSetDebugUtilsObjectNameEXT            SetDebugUtilsObjectNameEXT;
	static PFN_vkCmdBeginDebugUtilsLabelEXT            CmdBeginDebugUtilsLabelEXT;
	static PFN_vkCmdEndDebugUtilsLabelEXT              CmdEndDebugUtilsLabelEXT;
	static PFN_vkGetPhysicalDeviceMemoryProperties2KHR GetPhysicalDeviceMemoryProperties2KHR;

  public:
	// Extensions
	static const std::vector<const char *>                 s_extensions;
	static const std::vector<const char *>                 s_validation_layers;
	static const std::vector<VkValidationFeatureEnableEXT> s_validation_extensions;

  private:
	VkInstance m_handle = VK_NULL_HANDLE;

#ifdef _DEBUG
	bool m_debug_enable = true;
#else
	bool m_debug_enable = false;
#endif        // _DEBUG
};

// Vulkan Physical Device
class PhysicalDevice
{
  public:
	PhysicalDevice(VkInstance instance);
	~PhysicalDevice() = default;

	PhysicalDevice(const PhysicalDevice &) = delete;
	PhysicalDevice &operator=(const PhysicalDevice &) = delete;
	PhysicalDevice(PhysicalDevice &&)                 = delete;
	PhysicalDevice &operator=(PhysicalDevice &&) = delete;

	operator const VkPhysicalDevice &() const;

	const VkPhysicalDevice &                GetHandle() const;
	const VkPhysicalDeviceProperties &      GetProperties() const;
	const VkPhysicalDeviceFeatures &        GetFeatures() const;
	const VkPhysicalDeviceMemoryProperties &GetMemoryProperties() const;
	const VkSampleCountFlagBits &           GetSampleCount() const;

  private:
	VkPhysicalDevice                 m_handle            = VK_NULL_HANDLE;
	VkPhysicalDeviceProperties       m_properties        = {};
	VkPhysicalDeviceFeatures         m_features          = {};
	VkPhysicalDeviceMemoryProperties m_memory_properties = {};
	VkSampleCountFlagBits            m_max_samples_count = VK_SAMPLE_COUNT_1_BIT;
};

// Vulkan Surface
class Surface
{
  public:
	Surface(VkPhysicalDevice physical_device);
	~Surface();

	Surface(const Surface &) = delete;
	Surface &operator=(const Surface &) = delete;
	Surface(Surface &&)                 = delete;
	Surface &operator=(Surface &&) = delete;

	operator const VkSurfaceKHR &() const;

	const VkSurfaceKHR &            GetHandle() const;
	const VkSurfaceCapabilitiesKHR &GetCapabilities() const;
	const VkSurfaceFormatKHR &      GetFormat() const;

  private:
	VkSurfaceKHR             m_handle       = VK_NULL_HANDLE;
	VkSurfaceCapabilitiesKHR m_capabilities = {};
	VkSurfaceFormatKHR       m_format       = {};
};

// Vulkan Logical Device
class Device
{
  public:
	Device();
	~Device();

	Device(const Device &) = delete;
	Device &operator=(const Device &) = delete;
	Device(Device &&)                 = delete;
	Device &operator=(Device &&) = delete;

	operator const VkDevice &() const;

	const VkDevice &                GetHandle() const;
	const VkPhysicalDeviceFeatures &GetEnabledFeatures() const;
	const VmaAllocator &            GetAllocator() const;
	const uint32_t                  GetQueueFamily(QueueFamily queue) const;
	const Surface &                 GetSurface() const;
	const PhysicalDevice &          GetPhysicalDevice() const;
	VkQueue                         GetQueue(QueueFamily queue);

  public:
	static const std::vector<const char *> s_extensions;

  private:
	std::unique_ptr<PhysicalDevice> m_physical_device = nullptr;
	std::unique_ptr<Surface>        m_surface         = nullptr;

	VkDevice                 m_handle           = VK_NULL_HANDLE;
	VmaAllocator             m_allocator        = VK_NULL_HANDLE;
	VkQueueFlags             m_support_queues   = {};
	VkPhysicalDeviceFeatures m_enabled_features = {};

	std::map<QueueFamily, uint32_t>   m_queue_family;
	std::vector<uint32_t>             m_queue_index;
	std::vector<std::vector<VkQueue>> m_queues;
};

// Vulkan Swapchain
class Swapchain
{
  public:
	Swapchain(uint32_t width, uint32_t height, bool vsync = false, Swapchain *old_swapchain = nullptr);
	~Swapchain();

	Swapchain(const Swapchain &) = delete;
	Swapchain &operator=(const Swapchain &) = delete;
	Swapchain(Swapchain &&)                 = delete;
	Swapchain &operator=(Swapchain &&) = delete;

	operator const VkSwapchainKHR &() const;

	const VkSwapchainKHR &GetHandle() const;
	uint32_t              GetCurrentIndex() const;
	//std::shared_ptr<VKTexture> GetCurrentImage();
	uint32_t GetImageCount() const;

	void AcquireNextImage();
	void Present(VkSemaphore semaphore);

  private:
	uint32_t m_width;
	uint32_t m_height;
	bool     m_vsync;

	VkSwapchainKHR m_handle = VK_NULL_HANDLE;

	uint32_t m_current_index = 0;
	uint32_t m_image_count   = 0;
};

}        // namespace Ilum::Vulkan