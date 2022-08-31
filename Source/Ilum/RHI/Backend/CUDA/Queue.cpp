#include "Queue.hpp"
#include "Command.hpp"

namespace Ilum::CUDA
{
Queue::Queue(RHIDevice *device, RHIQueueFamily family, uint32_t queue_index) :
    RHIQueue(device, family, queue_index)
{
}

void Queue::Wait()
{

}

void Queue::Submit(const std::vector<RHICommand*>& cmds, const std::vector<RHISemaphore*>& signal_semaphores, const std::vector<RHISemaphore*>& wait_semaphores)
{
	for (auto& cmd : cmds)
	{
		m_cmds.push_back(cmd);
	}
}

void Queue::Execute(RHIFence* fence)
{
	for (auto& cmd : m_cmds)
	{
		static_cast<Command *>(cmd)->Execute();
	}
	m_cmds.clear();
}
}        // namespace Ilum::CUDA