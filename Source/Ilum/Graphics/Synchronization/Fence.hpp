#pragma once

#include "Utils/PCH.hpp"

namespace Ilum
{
class Fence
{
  public:
	Fence();

	~Fence();

	void wait() const;

	void reset() const;

	bool isSignaled() const;

	const VkFence &getFence() const;

	operator const VkFence &() const;

  private:
	VkFence m_handle = VK_NULL_HANDLE;
};
}        // namespace Ilum