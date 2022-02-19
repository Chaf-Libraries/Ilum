#pragma once

#include "Graphics/Vulkan.hpp"

#include <atomic>
#include <vector>

namespace Ilum::Graphics
{
class GLSLCompiler
{
  public:
	static std::vector<uint32_t> Compile(const std::vector<uint8_t> &data, VkShaderStageFlagBits stage);

  private:
	static std::atomic<uint32_t> s_task_count;
};
}