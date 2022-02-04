#pragma once

#include "../Vulkan.hpp"

#include <atomic>
#include <vector>

namespace Ilum::Vulkan
{
class GLSLCompiler
{
  public:
	static std::vector<uint32_t> Compile(const std::vector<uint8_t> &data, VkShaderStageFlagBits stage);

  private:
	static std::atomic<uint32_t> s_task_count;
};
}        // namespace Ilum::Vulkan