#include "Logger.hpp"

namespace Ilum::Vulkan
{
const bool check(VkResult result)
{
	if (result == VK_SUCCESS)
	{
		return true;
	}

	VK_ERROR("{}", std::to_string(result));
	return false;
}

void _assert(VkResult result)
{
	assert(result == VK_SUCCESS);
}
}        // namespace Ilum::Vulkan