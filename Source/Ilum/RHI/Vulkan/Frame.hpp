#pragma once

#include <volk.h>

#include <vector>
#include <unordered_map>
#include <thread>

namespace Ilum::Vulkan
{
class Frame
{
  public:
	Frame();
	~Frame();

  private:
	std::unordered_map<std::thread::id, VkCommandPool> m_cmd_pools;
};
}        // namespace Ilum::Vulkan