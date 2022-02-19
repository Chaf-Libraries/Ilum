#pragma once

#include <Graphics/Vulkan.hpp>

#include <vector>

namespace Ilum::Resource
{
struct Bitmap
{
	std::vector<uint8_t> data;
	VkFormat             format = VK_FORMAT_UNDEFINED;
	uint32_t             width  = 0;
	uint32_t             height = 0;
};
}        // namespace Ilum::Resource