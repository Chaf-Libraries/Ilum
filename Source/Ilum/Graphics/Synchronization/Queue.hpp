#pragma once

#include "Utils/PCH.hpp"

namespace Ilum
{
class CommandBuffer;

struct SubmitInfo
{
	VkSemaphore              signal_semaphore;
	std::vector<VkSemaphore> wait_semaphores;
	VkFence                  fence;
	VkPipelineStageFlags    stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
};

class Queue
{
  public:
	Queue(VkQueue handle);

	~Queue() = default;

	void submit(const CommandBuffer &command_buffer,
	            const VkSemaphore &  signal_semaphore = VK_NULL_HANDLE,
	            const VkSemaphore &  wait_semaphore   = VK_NULL_HANDLE,
	            const VkFence &      fence            = VK_NULL_HANDLE,
	            VkPipelineStageFlags wait_stages      = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT);

	void submit(const std::vector<CommandBuffer> &command_buffers,
	            const std::vector<VkSemaphore> &  signal_semaphores = {},
	            const std::vector<VkSemaphore> &  wait_semaphores   = {},
	            const VkFence &                   fence             = VK_NULL_HANDLE,
	            VkPipelineStageFlags                wait_stages       = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT);

	void submit(const CommandBuffer &command_buffer, const SubmitInfo &submit_info);

	void submitIdle(const CommandBuffer &command_buffer);

	void submitIdle(const std::vector<CommandBuffer> &command_buffers);

	void waitIdle();

	bool isBusy() const;

	const VkQueue &getQueue() const;

	operator const VkQueue &() const;

  private:
	VkQueue    m_handle = VK_NULL_HANDLE;
	std::mutex m_mutex;
	bool       m_busy = false;
};
}        // namespace Ilum