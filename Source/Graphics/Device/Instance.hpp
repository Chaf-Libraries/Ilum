#pragma once

#include "../Vulkan.hpp"

namespace Ilum::Graphics
{
class Instance
{
  public:
	Instance();
	~Instance();

	Instance(const Instance &) = delete;
	Instance &operator=(const Instance &) = delete;
	Instance(Instance &&)                 = delete;
	Instance &operator=(Instance &&) = delete;

	operator const VkInstance &() const;

	const VkInstance &GetHandle() const;

  private:
	VkInstance m_handle = VK_NULL_HANDLE;

#ifdef _DEBUG
	bool m_debug_enable = true;
#else
	bool m_debug_enable = false;
#endif        // _DEBUG
};
}        // namespace Ilum::Graphics