#pragma once

#include <volk.h>
#include <vk_mem_alloc.h>

#include <string>

#define VK_CHECK(result) vk_check(result)
#define VK_ASSERT(result) vk_assert(result)

namespace Ilum
{
const bool vk_check(VkResult result);

void vk_assert(VkResult result);

std::string shader_stage_to_string(VkShaderStageFlags stage);
}

namespace std
{
std::string to_string(VkResult result);

std::string to_string(VkFormat format);

std::string to_string(VkShaderStageFlagBits stage);
}