#include "Semaphore.hpp"

#include "Graphics/GraphicsContext.hpp"
#include "Graphics/Vulkan/Vulkan.hpp"

#include "Device/LogicalDevice.hpp"

namespace Ilum
{
Semaphore::Semaphore(bool timeline):
    m_timeline(timeline)
{
	VkSemaphoreTypeCreateInfo type_create_info{VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
	type_create_info.pNext         = nullptr;
	type_create_info.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
	type_create_info.initialValue  = 0;

	VkSemaphoreCreateInfo create_info{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
	create_info.pNext = m_timeline ? &type_create_info : nullptr;
	create_info.flags = 0;

	if (!VK_CHECK(vkCreateSemaphore(GraphicsContext::instance()->getLogicalDevice(), &create_info, nullptr, &m_handle)))
	{
		VK_ERROR("Failed to create semaphore");
	}
}

Semaphore::~Semaphore()
{
	if (m_handle)
	{
		vkDestroySemaphore(GraphicsContext::instance()->getLogicalDevice(), m_handle, nullptr);
	}
}

Semaphore::Semaphore(Semaphore &&other) noexcept :
    m_handle(other.m_handle)
{
	other.m_handle = VK_NULL_HANDLE;
}

Semaphore &Semaphore::operator=(Semaphore &&other) noexcept
{
	m_handle = other.m_handle;
	m_handle = VK_NULL_HANDLE;

	return *this;
}

bool Semaphore::wait(const uint64_t value, const uint64_t timeout) const
{
	assert(m_timeline);

	VkSemaphoreWaitInfo wait_info{VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
	wait_info.pNext          = nullptr;
	wait_info.flags          = 0;
	wait_info.semaphoreCount = 1;
	wait_info.pSemaphores    = &m_handle;
	wait_info.pValues        = &value;

	return VK_CHECK(vkWaitSemaphores(GraphicsContext::instance()->getLogicalDevice(), &wait_info, timeout));
}

bool Semaphore::signal(const uint64_t value) const
{
	assert(m_timeline);

	VkSemaphoreSignalInfo signal_info{VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO};
	signal_info.pNext     = nullptr;
	signal_info.semaphore = m_handle;
	signal_info.value     = value;

	return VK_CHECK(vkSignalSemaphore(GraphicsContext::instance()->getLogicalDevice(), &signal_info));
}

bool Semaphore::isTimeline() const
{
	return m_timeline;
}

uint64_t Semaphore::count() const
{
	assert(m_timeline);

	uint64_t value = 0;
	VK_CHECK(vkGetSemaphoreCounterValue(GraphicsContext::instance()->getLogicalDevice(), m_handle, &value));

	return value;
}

const VkSemaphore &Semaphore::getSemaphore() const
{
	return m_handle;
}

Semaphore::operator const VkSemaphore &() const
{
	return m_handle;
}
}        // namespace Ilum