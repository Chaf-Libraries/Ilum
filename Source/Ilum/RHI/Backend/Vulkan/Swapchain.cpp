#pragma once

#include "Swapchain.hpp"
#include "Command.hpp"
#include "Device.hpp"
#include "Queue.hpp"
#include "Synchronization.hpp"
#include "Texture.hpp"

#include <Core/Time.hpp>

#ifdef _WIN32
#	include <Windows.h>
#endif

namespace Ilum::Vulkan
{
Swapchain::Swapchain(RHIDevice *device, Window *window) :
    RHISwapchain(device, window)
{
#ifdef _WIN32
	{
		VkWin32SurfaceCreateInfoKHR createInfo{};
		createInfo.sType     = VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR;
		createInfo.hwnd      = (HWND) window->GetNativeHandle();
		createInfo.hinstance = GetModuleHandle(nullptr);
		vkCreateWin32SurfaceKHR(static_cast<Device *>(device)->GetInstance(), &createInfo, nullptr, &m_surface);
	}
#endif        // _WIN32

	// m_capabilities
	vkGetPhysicalDeviceSurfaceCapabilitiesKHR(static_cast<Device *>(device)->GetPhysicalDevice(), m_surface, &m_capabilities);

	// formats
	uint32_t                        format_count;
	std::vector<VkSurfaceFormatKHR> formats;
	vkGetPhysicalDeviceSurfaceFormatsKHR(static_cast<Device *>(device)->GetPhysicalDevice(), m_surface, &format_count, nullptr);
	if (format_count != 0)
	{
		formats.resize(format_count);
		vkGetPhysicalDeviceSurfaceFormatsKHR(static_cast<Device *>(device)->GetPhysicalDevice(), m_surface, &format_count, formats.data());
	}

	// present modes
	uint32_t                      presentmode_count;
	std::vector<VkPresentModeKHR> presentmodes;
	vkGetPhysicalDeviceSurfacePresentModesKHR(static_cast<Device *>(device)->GetPhysicalDevice(), m_surface, &presentmode_count, nullptr);
	if (presentmode_count != 0)
	{
		presentmodes.resize(presentmode_count);
		vkGetPhysicalDeviceSurfacePresentModesKHR(static_cast<Device *>(device)->GetPhysicalDevice(), m_surface, &presentmode_count, presentmodes.data());
	}

	// Choose swapchain surface format
	for (const auto &surface_format : formats)
	{
		if (surface_format.format == VK_FORMAT_B8G8R8A8_UNORM &&
		    surface_format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
		{
			m_surface_format = surface_format;
		}
	}
	if (m_surface_format.format == VK_FORMAT_UNDEFINED)
	{
		m_surface_format = formats[0];
	}

	// Choose swapchain present mode
	for (VkPresentModeKHR present_mode : presentmodes)
	{
		if (VK_PRESENT_MODE_MAILBOX_KHR == present_mode)
		{
			m_present_mode = VK_PRESENT_MODE_MAILBOX_KHR;
		}
	}
	if (m_present_mode == VK_PRESENT_MODE_MAX_ENUM_KHR)
	{
		m_present_mode = VK_PRESENT_MODE_FIFO_KHR;
	}

	// Choose swapchain extent
	VkExtent2D chosen_extent = {};
	if (m_capabilities.currentExtent.width != UINT32_MAX)
	{
		chosen_extent = m_capabilities.currentExtent;
	}
	else
	{
		VkExtent2D actualExtent = {};
		actualExtent.width      = p_window->GetWidth();
		actualExtent.height     = p_window->GetHeight();

		actualExtent.width =
		    std::clamp(actualExtent.width, m_capabilities.minImageExtent.width, m_capabilities.maxImageExtent.width);
		actualExtent.height =
		    std::clamp(actualExtent.height, m_capabilities.minImageExtent.height, m_capabilities.maxImageExtent.height);

		chosen_extent = actualExtent;
	}

	m_image_count = m_capabilities.minImageCount + 1;
	if (m_capabilities.maxImageCount > 0 && m_image_count > m_capabilities.maxImageCount)
	{
		m_image_count = m_capabilities.maxImageCount;
	}

	VkBool32 support = VK_FALSE;

	if (!support)
	{
		vkGetPhysicalDeviceSurfaceSupportKHR(static_cast<Device *>(p_device)->GetPhysicalDevice(), VK_QUEUE_GRAPHICS_BIT, m_surface, &support);
		if (support)
		{
			m_present_queue = std::make_unique<Queue>(device, RHIQueueFamily::Graphics);
		}
	}

	if (!support)
	{
		vkGetPhysicalDeviceSurfaceSupportKHR(static_cast<Device *>(p_device)->GetPhysicalDevice(), VK_QUEUE_COMPUTE_BIT, m_surface, &support);
		if (support)
		{
			m_present_queue = std::make_unique<Queue>(device, RHIQueueFamily::Compute);
		}
	}

	if (!support)
	{
		vkGetPhysicalDeviceSurfaceSupportKHR(static_cast<Device *>(p_device)->GetPhysicalDevice(), VK_QUEUE_TRANSFER_BIT, m_surface, &support);
		if (support)
		{
			m_present_queue = std::make_unique<Queue>(device, RHIQueueFamily::Transfer);
		}
	}

	CreateSwapchain(chosen_extent);
}

Swapchain::~Swapchain()
{
	if (m_swapchain)
	{
		vkDestroySwapchainKHR(static_cast<Device *>(p_device)->GetDevice(), m_swapchain, nullptr);
	}

	if (m_surface)
	{
		vkDestroySurfaceKHR(static_cast<Device *>(p_device)->GetInstance(), m_surface, nullptr);
	}
}

uint32_t Swapchain::GetTextureCount()
{
	return static_cast<uint32_t>(m_textures.size());
}

void Swapchain::AcquireNextTexture(RHISemaphore *signal_semaphore, RHIFence *signal_fence)
{
	auto result = vkAcquireNextImageKHR(
	    static_cast<Device *>(p_device)->GetDevice(),
	    m_swapchain,
	    std::numeric_limits<uint64_t>::max(),
	    signal_semaphore ? static_cast<Semaphore *>(signal_semaphore)->GetHandle() : nullptr,
	    signal_fence ? static_cast<Fence *>(signal_fence)->GetHandle() : nullptr,
	    &m_frame_index);
}

RHITexture *Swapchain::GetCurrentTexture()
{
	return m_textures[m_frame_index].get();
}

uint32_t Swapchain::GetCurrentFrameIndex()
{
	return m_frame_index;
}

bool Swapchain::Present(RHISemaphore *semaphore)
{
	VkSemaphore semaphore_handle = semaphore ? static_cast<Semaphore *>(semaphore)->GetHandle() : VK_NULL_HANDLE;

	VkPresentInfoKHR present_info = {};
	present_info.sType            = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
	present_info.pNext            = NULL;
	present_info.swapchainCount   = 1;
	present_info.pSwapchains      = &m_swapchain;
	present_info.pImageIndices    = &m_frame_index;

	if (semaphore)
	{
		present_info.pWaitSemaphores    = &semaphore_handle;
		present_info.waitSemaphoreCount = 1;
	}

	auto result = vkQueuePresentKHR(m_present_queue->GetHandle(), &present_info);

	if (result == VK_ERROR_OUT_OF_DATE_KHR)
	{
		CreateSwapchain(VkExtent2D{static_cast<uint32_t>(p_window->GetWidth()), static_cast<uint32_t>(p_window->GetHeight())});
		LOG_INFO("Swapchain resize to {} x {}", p_window->GetWidth(), p_window->GetHeight());
		m_frame_index = 0;
	}

	return result == VK_SUCCESS;
}

void Swapchain::CreateSwapchain(const VkExtent2D &extent)
{
	VkSwapchainCreateInfoKHR createInfo = {};
	createInfo.sType                    = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
	createInfo.surface                  = m_surface;

	createInfo.minImageCount    = m_image_count;
	createInfo.imageFormat      = m_surface_format.format;
	createInfo.imageColorSpace  = m_surface_format.colorSpace;
	createInfo.imageExtent      = extent;
	createInfo.imageArrayLayers = 1;
	createInfo.imageUsage       = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

	uint32_t queueFamilyIndices[] = {static_cast<Device *>(p_device)->GetQueueFamily(m_present_queue->GetQueueFamily())};

	createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
	createInfo.preTransform     = m_capabilities.currentTransform;
	createInfo.compositeAlpha   = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
	createInfo.presentMode      = m_present_mode;
	createInfo.clipped          = VK_TRUE;

	createInfo.oldSwapchain = m_swapchain;

	vkCreateSwapchainKHR(static_cast<Device *>(p_device)->GetDevice(), &createInfo, nullptr, &m_swapchain);

	{
		m_textures.clear();

		TextureDesc desc = {};
		desc.width       = extent.width;
		desc.height      = extent.height;
		desc.depth       = 1;
		desc.mips        = 1;
		desc.layers      = 1;
		desc.samples     = 1;

		// TODO: Maybe different format?
		desc.format = RHIFormat::B8G8R8A8_UNORM;
		desc.usage  = RHITextureUsage::RenderTarget;

		uint32_t m_image_count = 0;

		std::vector<VkImage> images;
		vkGetSwapchainImagesKHR(static_cast<Device *>(p_device)->GetDevice(), m_swapchain, &m_image_count, nullptr);
		images.resize(m_image_count);
		vkGetSwapchainImagesKHR(static_cast<Device *>(p_device)->GetDevice(), m_swapchain, &m_image_count, images.data());

		for (size_t i = 0; i < images.size(); i++)
		{
			m_textures.emplace_back(std::make_unique<Texture>(p_device, desc, images[i]));
		}
	}

	m_frame_index = 0;
}
}        // namespace Ilum::Vulkan