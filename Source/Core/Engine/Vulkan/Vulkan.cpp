#define VMA_IMPLEMENTATION
#define VOLK_IMPLEMENTATION

#include "Vulkan.hpp"

#include "Core/Engine/Logging/Logger.hpp"
#include "Core/Device/Instance.hpp"

namespace Ilum
{
const bool vk_check(VkResult result)
{
	if (result == VK_SUCCESS)
	{
		return true;
	}

	VK_ERROR("{}", std::to_string(result));
	return false;
}

void vk_assert(VkResult result)
{
#ifdef _DEBUG
	assert(result == VK_SUCCESS);
#else
	VK_ERROR("{}", std::to_string(result));
#endif        // _DEBUG
}
}        // namespace Ilum::Vulkan