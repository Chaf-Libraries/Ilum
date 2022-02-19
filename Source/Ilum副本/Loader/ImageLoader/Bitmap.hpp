#pragma once

#include "Utils/PCH.hpp"

namespace Ilum
{
struct Bitmap
{
	std::vector<uint8_t> data;
	VkFormat             format = VK_FORMAT_UNDEFINED;
	uint32_t             width  = 0;
	uint32_t             height = 0;
	std::vector<std::vector<uint8_t>> mip_levels;
};

struct Cubemap
{
	std::array<std::vector<uint8_t>, 6> data;
	VkFormat                            format = VK_FORMAT_UNDEFINED;
	uint32_t                            width  = 0;
	uint32_t                            height = 0;
};
}        // namespace Ilum