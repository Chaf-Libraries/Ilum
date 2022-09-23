#include "Queue.hpp"
#include "Command.hpp"
#include "Synchronization.hpp"

#include <cuda.h>
#include <cuda_runtime.h>

namespace Ilum::CUDA
{
Queue::Queue(RHIDevice *device) :
    RHIQueue(device)
{
}

void Queue::Wait()
{
	cudaDeviceSynchronize();
}

void Queue::Execute(RHIQueueFamily family, const std::vector<SubmitInfo> &submit_infos, RHIFence *fence)
{
	for (auto &submit_info : submit_infos)
	{
		for (auto &wait_semaphore : submit_info.wait_semaphores)
		{
			static_cast<Semaphore *>(wait_semaphore)->Wait();
		}
		for (auto &cmd_buffer : submit_info.cmd_buffers)
		{
			static_cast<Command *>(cmd_buffer)->Execute();
		}
		for (auto &signal_semaphore : submit_info.signal_semaphores)
		{
			static_cast<Semaphore *>(signal_semaphore)->Wait();
		}
	}
}

void Queue::Execute(RHICommand *cmd_buffer)
{
	static_cast<Command *>(cmd_buffer)->Execute();
	cudaDeviceSynchronize();
}
}        // namespace Ilum::CUDA