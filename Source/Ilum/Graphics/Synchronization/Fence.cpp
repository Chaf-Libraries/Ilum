#include "Fence.hpp"

#include "Graphics/GraphicsContext.hpp"

#include <Graphics/Device/Device.hpp>
#include <Graphics/RenderContext.hpp>

namespace Ilum
{
Fence::Fence()
{
	VkFenceCreateInfo fence_create_info = {};
	fence_create_info.sType             = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
	fence_create_info.flags             = 0;

	vkCreateFence(Graphics::RenderContext::GetDevice(), &fence_create_info, nullptr, &m_handle);
}

Fence::~Fence()
{
	if (m_handle)
	{
		vkDestroyFence(Graphics::RenderContext::GetDevice(), m_handle, nullptr);
	}
}

Fence::Fence(Fence &&other) noexcept:
    m_handle(other.m_handle)
{
	other.m_handle = VK_NULL_HANDLE;
}

Fence &Fence::operator=(Fence &&other) noexcept
{
	m_handle = other.m_handle;
	other.m_handle = VK_NULL_HANDLE;

	return *this;
}

void Fence::wait() const
{
	vkWaitForFences(Graphics::RenderContext::GetDevice(), 1, &m_handle, VK_TRUE, std::numeric_limits<uint64_t>::max());
}

void Fence::reset() const
{
	vkResetFences(Graphics::RenderContext::GetDevice(), 1, &m_handle);
}

bool Fence::isSignaled() const
{
	auto result = vkGetFenceStatus(Graphics::RenderContext::GetDevice(), m_handle);

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