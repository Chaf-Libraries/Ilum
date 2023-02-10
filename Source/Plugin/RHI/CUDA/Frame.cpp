#include "Frame.hpp"
#include "Command.hpp"

namespace Ilum::CUDA
{
Frame::Frame(RHIDevice *device):
    RHIFrame(device)
{
}

RHIFence *Frame::AllocateFence()
{
	return nullptr;
}

RHISemaphore *Frame::AllocateSemaphore()
{
	return nullptr;
}

RHICommand *Frame::AllocateCommand(RHIQueueFamily family)
{
	while (m_current_cmd >= m_cmds.size())
	{
		m_cmds.emplace_back(std::make_unique<Command>(p_device, family));
	}

	return m_cmds[m_current_cmd++].get();
}

void Frame::Reset()
{
	m_current_cmd = 0;
}
}        // namespace Ilum::CUDA