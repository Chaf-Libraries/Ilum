#pragma once

#include "RHI/RHIQueue.hpp"

#include <volk.h>

namespace Ilum::Vulkan
{
class Device;

class Queue : public RHIQueue
{
  public:
	Queue(RHIDevice *device);

	virtual ~Queue();

	virtual void Execute(RHIQueueFamily family, const std::vector<SubmitInfo> &submit_infos, RHIFence *fence) override;

	virtual void Execute(RHICommand *cmd_buffer) override;

	virtual void Wait() override;

	VkQueue GetHandle(RHIQueueFamily family, uint32_t index) const;

  private:
	Device *p_device = nullptr;

	std::map<RHIQueueFamily, std::vector<VkQueue>>  m_queues;
	std::map<RHIQueueFamily, std::atomic<size_t>> m_queue_index;

	std::unordered_map<VkQueue, VkFence> m_queue_fences;
};
}        // namespace Ilum::Vulkan