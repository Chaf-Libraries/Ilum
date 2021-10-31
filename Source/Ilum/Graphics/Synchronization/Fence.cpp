#include "Fence.hpp"

#include "Graphics/GraphicsContext.hpp"

#include"Device/LogicalDevice.hpp"

namespace Ilum
{
Fence::Fence()
{
	VkFenceCreateInfo fence_create_info = {};
	fence_create_info.sType             = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
	fence_create_info.flags             = 0;

	vkCreateFence(GraphicsContext::instance()->getLogicalDevice(), &fence_create_info, nullptr, &m_handle);
}

Fence::~Fence()
{
	if (m_handle)
	{
		vkDestroyFence(GraphicsContext::instance()->getLogicalDevice(), m_handle, nullptr);
	}
}

void Fence::wait() const
{
	vkWaitForFences(GraphicsContext::instance()->getLogicalDevice(), 1, &m_handle, VK_TRUE, std::numeric_limits<uint64_t>::max());
}

void Fence::reset() const
{
	vkResetFences(GraphicsContext::instance()->getLogicalDevice(), 1, &m_handle);
}

bool Fence::isSignaled() const
{
	auto result = vkGetFenceStatus(GraphicsContext::instance()->getLogicalDevice(), m_handle);

	if (result == VK_SUCCESS)
	{
		return true;
	}

	return false;
}

const VkFence &Fence::getFence() const
{
	return m_handle;
}

Fence::operator const VkFence &() const
{
	return m_handle;
}
}        // namespace Ilum