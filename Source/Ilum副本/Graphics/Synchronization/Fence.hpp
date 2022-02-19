#pragma once

#include "Utils/PCH.hpp"

namespace Ilum
{
class Fence
{
  public:
	Fence();

	~Fence();

	Fence(const Fence &) = delete;

	Fence &operator=(const Fence &) = delete;

	Fence(Fence &&other) noexcept;

	Fence &operator=(Fence &&other) noexcept;

	void wait() const;

	void reset() const;

	bool isSignaled() const;

	const VkFence &getFence() const;

	operator const VkFence &() const;

  private:
	VkFence m_handle = VK_NULL_HANDLE;
};
}        // namespace Ilum