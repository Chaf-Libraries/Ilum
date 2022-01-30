#include "VKCommandPool.hpp"
#include "VKContext.hpp"
#include "VKDevice.hpp"

#include <Core/Hash.hpp>

namespace Ilum::RHI::Vulkan
{
VKCommandPool::VKCommandPool(const CmdUsage &usage, const std::thread::id &thread_id) :
    m_usage(usage),
    m_thread_id(thread_id)
{
	m_hash = 0;
	Core::HashCombine(m_hash, static_cast<size_t>(usage));
	Core::HashCombine(m_hash, thread_id);

	uint32_t queue_family = 0;

	switch (m_usage)
	{
		case Ilum::RHI::CmdUsage::Graphics:
			queue_family = VKContext::GetDevice().GetGraphicsFamily();
			break;
		case Ilum::RHI::CmdUsage::Compute:
			queue_family = VKContext::GetDevice().GetComputeFamily();
			break;
		case Ilum::RHI::CmdUsage::Transfer:
			queue_family = VKContext::GetDevice().GetTransferFamily();
			break;
		default:
			break;
	}

	VkCommandPoolCreateInfo create_info = {};
	create_info.sType                   = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
	create_info.flags                   = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
	create_info.queueFamilyIndex        = queue_family;

	vkCreateCommandPool(VKContext::GetDevice(), &create_info, nullptr, &m_handle);
}

VKCommandPool::~VKCommandPool()
{
	if (m_handle)
	{
		vkDestroyCommandPool(VKContext::GetDevice(), m_handle, nullptr);
	}
}

VKCommandPool::operator const VkCommandPool &() const
{
	return m_handle;
}

const VkCommandPool &VKCommandPool::GetHandle() const
{
	return m_handle;
}

void VKCommandPool::Reset()
{
	if (m_handle)
	{
		vkResetCommandPool(VKContext::GetDevice(), m_handle, 0);
	}
}

const std::thread::id &VKCommandPool::GetThreadID() const
{
	return m_thread_id;
}

const CmdUsage &VKCommandPool::GetUsage() const
{
	return m_usage;
}

size_t VKCommandPool::GetHash() const
{
	return m_hash;
}
}        // namespace Ilum::RHI::Vulkan