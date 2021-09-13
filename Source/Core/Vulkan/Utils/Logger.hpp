#pragma once

#include "Core/Engine/Logging/Logger.hpp"

#include <volk.h>

#define VK_INFO(...) Ilum::Logger::getInstance().log("vulkan", spdlog::level::info, __VA_ARGS__);
#define VK_WARN(...) Ilum::Logger::getInstance().log("vulkan", spdlog::level::warn, __VA_ARGS__);
#define VK_ERROR(...) Ilum::Logger::getInstance().log("vulkan", spdlog::level::err, __VA_ARGS__);
#define VK_TRACE(...) Ilum::Logger::getInstance().log("vulkan", spdlog::level::trace, __VA_ARGS__);
#define VK_CRITICAL(...) Ilum::Logger::getInstance().log("vulkan", spdlog::level::critical, __VA_ARGS__);
#define VK_DEBUG(x, ...) Ilum::Logger::getInstance().debug("vulkan", x, __VA_ARGS__);
#define VK_CHECK(result) Ilum::Vulkan::check(result)
#define VK_ASSERT(result) Ilum::Vulkan::_assert(result)

namespace Ilum::Vulkan
{
const bool check(VkResult result);
void       _assert(VkResult result);
}        // namespace Ilum::Vulkan