#pragma once

#include "Vulkan.hpp"

namespace Ilum::Vulkan
{
class Fence
{
  public:
	Fence();
	~Fence();

	Fence(const Fence &) = delete;
	Fence &operator=(const Fence &) = delete;
	Fence(Fence &&)                 = delete;
	Fence &operator=(Fence &&) = delete;

	void           Wait() const;
	void           Reset() const;
	bool           IsSignaled() const;
	const VkFence &GetHandle() const;

	operator const VkFence &() const;

  private:
	VkFence m_handle = VK_NULL_HANDLE;
};
}        // namespace Ilum::Vulkan