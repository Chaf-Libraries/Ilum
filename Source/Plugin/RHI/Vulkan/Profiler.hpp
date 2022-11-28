#pragma once

#include "Fwd.hpp"

namespace Ilum::Vulkan
{
class Profiler : public RHIProfiler
{
  public:
	Profiler(RHIDevice *device, uint32_t frame_count);

	virtual ~Profiler() override;

	virtual void Begin(RHICommand *cmd_buffer, uint32_t frame_index) override;

	virtual void End(RHICommand *cmd_buffer) override;

  private:
	std::vector<VkQueryPool> m_query_pools;
	uint32_t                 m_current_index = 0;
	VkCommandBuffer          m_cmd_buffer    = VK_NULL_HANDLE;
};
}        // namespace Ilum::Vulkan