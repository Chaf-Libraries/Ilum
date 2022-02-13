#pragma once

#include "../Vulkan.hpp"

namespace Ilum::Graphics
{
class Device;
class Surface;
class PhysicalDevice;

class Swapchain
{
  public:
	Swapchain(const Device &device, const Surface &surface, const PhysicalDevice &physical_device, VkSwapchainKHR old_swapchain = VK_NULL_HANDLE, bool vsync = false);
	~Swapchain();

	operator const VkSwapchainKHR &() const;

	const VkSwapchainKHR &GetHandle() const;
	const VkExtent2D &    GetExtent() const;
	uint32_t              GetImageCount() const;
	const VkImage &       GetActiveImage() const;

	VkResult AcquireNextImage(VkSemaphore image_avaliable_semaphore);
	VkResult Present(VkSemaphore wait_semaphore);

  private:
	const Device& m_device;

	VkExtent2D                  m_extent          = {};
	VkPresentModeKHR            m_present_mode    = {};
	VkSurfaceTransformFlagsKHR  m_pre_transform   = {};
	VkCompositeAlphaFlagBitsKHR m_composite_alpha = {};
	VkSwapchainKHR              m_handle          = VK_NULL_HANDLE;

	std::vector<VkImage> m_images;

	uint32_t m_active_image_index = 0;
};
}        // namespace Ilum::Graphics