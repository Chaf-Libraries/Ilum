#include "Synchronize.hpp"
#include "Device.hpp"
#include "RenderContext.hpp"

namespace Ilum::Vulkan
{
Fence::Fence()
{
	VkFenceCreateInfo create_info = {};
	create_info.sType             = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
	create_info.flags             = 0;

	vkCreateFence(RenderContext::GetDevice(), &create_info, nullptr, &m_handle);
}

Fence::~Fence()
{
	if (m_handle)
	{
		vkDestroyFence(RenderContext::GetDevice(), m_handle, nullptr);
	}
}

void Fence::Wait() const
{
	vkWaitForFences(RenderContext::GetDevice(), 1, &m_handle, VK_TRUE, std::numeric_limits<uint64_t>::max());
}

void Fence::Reset() const
{
	vkResetFences(RenderContext::GetDevice(), 1, &m_handle);
}

bool Fence::IsSignaled() const
{
	if (vkGetFenceStatus(RenderContext::GetDevice(), m_handle) == VK_SUCCESS)
	{
		return true;
	}

	return false;
}

const VkFence &Fence::GetHandle() const
{
	return m_handle;
}

Fence::operator const VkFence &() const
{
	return m_handle;
}
}        // namespace Ilum::Vulkan