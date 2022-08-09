#include "Frame.hpp"
#include "Command.hpp"
#include "Device.hpp"
#include "Synchronization.hpp"

namespace Ilum::Vulkan
{
Frame::Frame(RHIDevice *device) :
    RHIFrame(device)
{
}

Frame::~Frame()
{
	p_device->WaitIdle();

	m_fences.clear();
	m_semaphores.clear();
	m_commands.clear();

	for (auto &[hash, pool] : m_command_pools)
	{
		vkDestroyCommandPool(static_cast<Device *>(p_device)->GetDevice(), pool, nullptr);
	}

	m_command_pools.clear();
}

RHIFence *Frame::AllocateFence()
{
	if (m_fences.size() > m_active_fence_index)
	{
		return m_fences[m_active_fence_index].get();
	}

	while (m_fences.size() <= m_active_fence_index)
	{
		m_fences.emplace_back(std::make_unique<Fence>(p_device));
	}

	m_active_fence_index++;

	return m_fences.back().get();
}

RHISemaphore *Frame::AllocateSemaphore()
{
	if (m_semaphores.size() > m_active_semaphore_index)
	{
		return m_semaphores[m_active_semaphore_index].get();
	}

	while (m_semaphores.size() <= m_active_semaphore_index)
	{
		m_semaphores.emplace_back(std::make_unique<Semaphore>(p_device));
	}

	m_active_semaphore_index++;

	return m_semaphores.back().get();
}

RHICommand *Frame::AllocateCommand(RHIQueueFamily family)
{
	size_t hash = 0;
	HashCombine(hash, family, std::this_thread::get_id());

	if (m_command_pools.find(hash) == m_command_pools.end())
	{
		VkCommandPoolCreateInfo create_info = {};
		create_info.sType                   = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
		create_info.flags                   = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
		create_info.queueFamilyIndex        = static_cast<Device *>(p_device)->GetQueueFamily(family);

		VkCommandPool pool = VK_NULL_HANDLE;
		vkCreateCommandPool(static_cast<Device *>(p_device)->GetDevice(), &create_info, nullptr, &pool);

		m_command_pools.emplace(hash, pool);
	}

	if (m_commands.find(hash) == m_commands.end())
	{
		m_commands.emplace(hash, std::vector<std::unique_ptr<Command>>{});
	}

	if (m_commands[hash].size() > m_active_cmd_index[hash])
	{
		auto &cmd = m_commands[hash][m_active_cmd_index[hash]];
		cmd->Init();
		return cmd.get();
	}

	while (m_commands[hash].size() <= m_active_cmd_index[hash])
	{
		m_commands[hash].emplace_back(std::make_unique<Command>(p_device, m_command_pools[hash], family));
	}

	m_active_cmd_index[hash]++;

	auto &cmd = m_commands[hash].back();
	cmd->Init();
	return cmd.get();
}

void Frame::Reset()
{
	std::vector<VkFence> fences;

	fences.reserve(m_fences.size());

	for (auto &fence : m_fences)
	{
		fences.push_back(fence->GetHandle());
	}

	if (!fences.empty())
	{
		vkWaitForFences(static_cast<Device *>(p_device)->GetDevice(), static_cast<uint32_t>(fences.size()), fences.data(), VK_TRUE, UINT64_MAX);
		vkResetFences(static_cast<Device *>(p_device)->GetDevice(), static_cast<uint32_t>(fences.size()), fences.data());
	}

	for (auto &[hash, pool] : m_command_pools)
	{
		vkResetCommandPool(static_cast<Device *>(p_device)->GetDevice(), pool, 0);
		m_active_cmd_index[hash] = 0;
		for (auto &cmd : m_commands[hash])
		{
			cmd->SetState(CommandState::Available);
		}
	}

	m_active_fence_index     = 0;
	m_active_semaphore_index = 0;
}
}        // namespace Ilum::Vulkan