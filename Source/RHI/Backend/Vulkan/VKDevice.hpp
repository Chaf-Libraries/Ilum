#pragma once

#include "Vulkan.hpp"

#include "../../RHICommand.hpp"

#include <vector>

namespace Ilum::RHI::Vulkan
{
class VKSurface;
class VKCommandPool;

class VKPhysicalDevice
{
  public:
	VKPhysicalDevice(VkInstance instance);

	~VKPhysicalDevice() = default;

	operator const VkPhysicalDevice &() const;

	const VkPhysicalDevice &GetHandle() const;

	const VkPhysicalDeviceProperties &GetProperties() const;

	const VkPhysicalDeviceFeatures &GetFeatures() const;

	const VkPhysicalDeviceMemoryProperties &GetMemoryProperties() const;

	const VkSampleCountFlagBits &GetSampleCount() const;

  private:
	VkPhysicalDevice                 m_handle            = VK_NULL_HANDLE;
	VkPhysicalDeviceProperties       m_properties        = {};
	VkPhysicalDeviceFeatures         m_features          = {};
	VkPhysicalDeviceMemoryProperties m_memory_properties = {};
	VkSampleCountFlagBits            m_max_samples_count = VK_SAMPLE_COUNT_1_BIT;
};

class VKDevice
{
  public:
	VKDevice();

	~VKDevice();

	operator const VkDevice &() const;

	const VkDevice &GetHandle() const;

	const VkPhysicalDeviceFeatures &GetEnabledFeatures() const;

	const VmaAllocator &GetAllocator() const;

	const uint32_t GetGraphicsFamily() const;

	const uint32_t GetComputeFamily() const;

	const uint32_t GetTransferFamily() const;

	const uint32_t GetPresentFamily() const;

	VKCommandPool &AcquireCommandPool(const CmdUsage &usage, const std::thread::id &thread_id);

	VKCommandPool &GetCommandPool(size_t pool_index);

	bool HasCommandPool(size_t pool_index);

  public:
	static const std::vector<const char *> s_extensions;

  private:
	std::unique_ptr<VKPhysicalDevice> m_physical_device = nullptr;

	std::unique_ptr<VKSurface> m_surface = nullptr;

	std::unordered_map<size_t, std::unique_ptr<VKCommandPool>> m_command_pools;

	VkDevice m_handle = VK_NULL_HANDLE;

	VkPhysicalDeviceFeatures m_enabled_features = {};

	VmaAllocator m_allocator = VK_NULL_HANDLE;

	VkQueueFlags m_support_queues = {};

	uint32_t m_graphics_family = 0;
	uint32_t m_compute_family  = 0;
	uint32_t m_transfer_family = 0;
	uint32_t m_present_family  = 0;
};
}        // namespace Ilum::RHI::Vulkan