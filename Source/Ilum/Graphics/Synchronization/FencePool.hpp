#pragma once

#include "Utils/PCH.hpp"

namespace Ilum
{
class FencePool
{
  public:
	FencePool() = default;
	~FencePool();

	FencePool(const FencePool &) = delete;
	FencePool &operator=(const FencePool &) = delete;
	FencePool(FencePool &&)                 = delete;
	FencePool &operator=(FencePool &&) = delete;

	VkFence &requestFence();

	void wait(uint32_t timeout = std::numeric_limits<uint32_t>::max()) const;

	void reset();

  private:
	std::vector<VkFence> m_fences;

	uint32_t m_active_fence_count = 0;

	std::mutex m_mutex;
};
}        // namespace Ilum