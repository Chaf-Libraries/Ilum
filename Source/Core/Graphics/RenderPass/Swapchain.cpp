#include "Swapchain.hpp"

#include "Core/Device/LogicalDevice.hpp"
#include "Core/Device/PhysicalDevice.hpp"
#include "Core/Device/Surface.hpp"
#include "Core/Graphics/Image/Image.hpp"

namespace Ilum
{
Swapchain::Swapchain(const LogicalDevice &logical_device, const VkExtent2D &extent, const Swapchain *old_swapchain) :
    m_surface(logical_device.getSurface()),
    m_physical_device(logical_device.getPhysicalDevice()),
    m_logical_device(logical_device),
    m_extent(extent),
    m_present_mode(VK_PRESENT_MODE_FIFO_KHR),
    m_pre_transform(VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR),
    m_composite_alpha(VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR),
    m_active_image_index(std::numeric_limits<uint32_t>::max())
{
	auto &surface_format       = m_surface.getFormat();
	auto &surface_capabilities = m_surface.getCapabilities();
	auto  graphics_family      = m_logical_device.getGraphicsFamily();
	auto  present_family       = m_logical_device.getPresentFamily();

	// Get present mode
	uint32_t present_mode_count;
	vkGetPhysicalDeviceSurfacePresentModesKHR(m_physical_device, m_surface, &present_mode_count, nullptr);
	std::vector<VkPresentModeKHR> present_modes(present_mode_count);
	vkGetPhysicalDeviceSurfacePresentModesKHR(m_physical_device, m_surface, &present_mode_count, present_modes.data());

	for (const auto &present_mode : present_modes)
	{
		if (present_mode == VK_PRESENT_MODE_MAILBOX_KHR)
		{
			m_present_mode = present_mode;
			break;
		}

		else if (present_mode == VK_PRESENT_MODE_IMMEDIATE_KHR)
		{
			m_present_mode = present_mode;
		}
	}

	// Get swapchain image count
	auto desired_image_count = surface_capabilities.minImageCount + 1;

	if (surface_capabilities.maxImageCount > 0 && desired_image_count > surface_capabilities.maxImageCount)
	{
		desired_image_count = surface_capabilities.maxImageCount;
	}

	// Get pre transform support
	if (surface_capabilities.supportedTransforms & VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR)
	{
		m_pre_transform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR;
	}
	else
	{
		m_pre_transform = surface_capabilities.currentTransform;
	}

	// Get composite alpha
	const std::vector<VkCompositeAlphaFlagBitsKHR> composite_alpha_flags = {
	    VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
	    VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR,
	    VK_COMPOSITE_ALPHA_POST_MULTIPLIED_BIT_KHR,
	    VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR};

	for (const auto &composite_alpha_flag : composite_alpha_flags)
	{
		if (surface_capabilities.supportedCompositeAlpha & composite_alpha_flag)
		{
			m_composite_alpha = composite_alpha_flag;
			break;
		}
	}

	// Create swapchain
	VkSwapchainCreateInfoKHR swapchain_create_info = {};
	swapchain_create_info.sType                    = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
	swapchain_create_info.surface                  = m_surface;
	swapchain_create_info.minImageCount            = desired_image_count;
	swapchain_create_info.imageFormat              = surface_format.format;
	swapchain_create_info.imageColorSpace          = surface_format.colorSpace;
	swapchain_create_info.imageExtent              = m_extent;
	swapchain_create_info.imageArrayLayers         = 1;
	swapchain_create_info.imageUsage               = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
	swapchain_create_info.imageSharingMode         = VK_SHARING_MODE_EXCLUSIVE;
	swapchain_create_info.preTransform             = static_cast<VkSurfaceTransformFlagBitsKHR>(m_pre_transform);
	swapchain_create_info.compositeAlpha           = m_composite_alpha;
	swapchain_create_info.presentMode              = m_present_mode;
	swapchain_create_info.clipped                  = VK_TRUE;
	swapchain_create_info.oldSwapchain             = old_swapchain ? old_swapchain->m_handle : VK_NULL_HANDLE;
	swapchain_create_info.imageUsage |= surface_capabilities.supportedUsageFlags & VK_IMAGE_USAGE_TRANSFER_SRC_BIT ? VK_IMAGE_USAGE_TRANSFER_SRC_BIT : 0;
	swapchain_create_info.imageUsage |= surface_capabilities.supportedUsageFlags & VK_IMAGE_USAGE_TRANSFER_DST_BIT ? VK_IMAGE_USAGE_TRANSFER_DST_BIT : 0;

	if (graphics_family != present_family)
	{
		std::array<uint32_t, 2> queue_family        = {graphics_family, present_family};
		swapchain_create_info.imageSharingMode      = VK_SHARING_MODE_CONCURRENT;
		swapchain_create_info.queueFamilyIndexCount = static_cast<uint32_t>(queue_family.size());
		swapchain_create_info.pQueueFamilyIndices   = queue_family.data();
	}

	if (!VK_CHECK(vkCreateSwapchainKHR(m_logical_device, &swapchain_create_info, nullptr, &m_handle)))
	{
		VK_ERROR("Failed to create swapchain");
		return;
	}

	// Get swapchain images
	if (!VK_CHECK(vkGetSwapchainImagesKHR(m_logical_device, m_handle, &m_image_count, nullptr)))
	{
		VK_ERROR("Failed to get swapchain images");
		return;
	}

	m_images.resize(m_image_count);
	m_image_views.resize(m_image_count);

	if (!VK_CHECK(vkGetSwapchainImagesKHR(m_logical_device, m_handle, &m_image_count, m_images.data())))
	{
		VK_ERROR("Failed to get swapchain images");
		return;
	}

	for (uint32_t i = 0; i < m_image_count; i++)
	{
		Image::createImageView(m_logical_device, m_images[i], m_image_views[i], VK_IMAGE_VIEW_TYPE_2D, surface_format.format, 1, 0, 1, 0, VK_IMAGE_ASPECT_COLOR_BIT);
	}
}

Swapchain::~Swapchain()
{
	if (m_handle)
	{
		vkDestroySwapchainKHR(m_logical_device, m_handle, nullptr);
	}

	for (const auto &image_view : m_image_views)
	{
		if (image_view)
		{
			vkDestroyImageView(m_logical_device, image_view, nullptr);
		}
	}
}

VkResult Swapchain::acquireNextImage(const VkSemaphore &present_complete_semaphore, VkFence fence)
{
	if (fence != VK_NULL_HANDLE)
	{
		vkWaitForFences(m_logical_device, 1, &fence, VK_TRUE, std::numeric_limits<uint64_t>::max());
	}

	auto result = vkAcquireNextImageKHR(m_logical_device, m_handle, std::numeric_limits<uint64_t>::max(), present_complete_semaphore, VK_NULL_HANDLE, &m_active_image_index);

	if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR && result != VK_ERROR_OUT_OF_DATE_KHR)
	{
		throw std::runtime_error("Failed to acquire swapchain image!");
	}

	return result;
}

VkResult Swapchain::present(const VkQueue &present_queue, const VkSemaphore &wait_semaphore)
{
	VkPresentInfoKHR present_info   = {};
	present_info.sType              = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
	present_info.waitSemaphoreCount = 1;
	present_info.pWaitSemaphores    = &wait_semaphore;
	present_info.swapchainCount     = 1;
	present_info.pSwapchains        = &m_handle;
	present_info.pImageIndices      = &m_active_image_index;

	return vkQueuePresentKHR(present_queue, &present_info);
}

Swapchain::operator const VkSwapchainKHR &() const
{
	return m_handle;
}

const VkExtent2D &Swapchain::getExtent() const
{
	return m_extent;
}

uint32_t Swapchain::getImageCount() const
{
	return m_image_count;
}

const std::vector<VkImage> &Swapchain::getImages() const
{
	return m_images;
}

const VkImage &Swapchain::getActiveImage() const
{
	return m_images[m_active_image_index];
}

const std::vector<VkImageView> &Swapchain::getImageViews() const
{
	return m_image_views;
}

const VkSwapchainKHR &Swapchain::getSwapchain() const
{
	return m_handle;
}

uint32_t Swapchain::getActiveImageIndex() const
{
	return m_active_image_index;
}
}        // namespace Ilum