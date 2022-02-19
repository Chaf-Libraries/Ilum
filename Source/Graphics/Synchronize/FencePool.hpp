#pragma once

#include "Graphics/Vulkan.hpp"

namespace Ilum::Graphics
{
class Device;

class FencePool
{
  public:
	FencePool(const Device &device);
	~FencePool();

	FencePool(const FencePool &) = delete;
	FencePool &operator=(const FencePool &) = delete;
	FencePool(FencePool &&)                 = delete;
	FencePool &operator=(FencePool &&) = delete;

	VkFence &RequestFence();

	void Wait(uint32_t timeout = std::numeric_limits<uint32_t>::max()) const;

	void Reset();

  private:
	const Device &m_device;
	  
	std::vector<VkFence> m_fences;

	uint32_t m_active_fence_count = 0;

	std::mutex m_mutex;
};
}