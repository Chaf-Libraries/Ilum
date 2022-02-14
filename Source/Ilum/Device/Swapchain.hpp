#pragma once

#include "Utils/PCH.hpp"

#include <Graphics/Resource/Image.hpp>

namespace Ilum
{
class Surface;
class PhysicalDevice;
class LogicalDevice;

class Swapchain
{
  public:
	Swapchain(const VkExtent2D &extent, const Swapchain *old_swapchain = nullptr, bool vsync = false);

	~Swapchain();

	VkResult acquireNextImage(const VkSemaphore &present_complete_semaphore = VK_NULL_HANDLE, VkFence fence = VK_NULL_HANDLE);

	VkResult present(const VkQueue &present_queue, const VkSemaphore &wait_semaphore = VK_NULL_HANDLE);

	operator const VkSwapchainKHR &() const;

	const VkExtent2D &getExtent() const;

	uint32_t getImageCount() const;

	const std::vector<Graphics::Image> &getImages() const;

	const VkImage &getActiveImage() const;

	const VkSwapchainKHR &getSwapchain() const;

	uint32_t getActiveImageIndex() const;

	bool isVsync() const;

  private:
	VkExtent2D       m_extent       = {};
	VkPresentModeKHR m_present_mode = {};
	VkSurfaceTransformFlagsKHR m_pre_transform = {};
	VkCompositeAlphaFlagBitsKHR m_composite_alpha = {};

	uint32_t                 m_image_count = 0;
	std::vector<Graphics::Image> m_images;

	VkSwapchainKHR m_handle = VK_NULL_HANDLE;

	uint32_t m_active_image_index = 0;

	bool m_vsync = false;
};
}        // namespace Ilum