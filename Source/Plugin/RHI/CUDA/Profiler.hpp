#pragma once

#include "Fwd.hpp"

namespace Ilum::CUDA
{
class Profiler : public RHIProfiler
{
  public:
	Profiler(RHIDevice *device, uint32_t frame_count);

	virtual ~Profiler() override;

	virtual void Begin(RHICommand *cmd_buffer, uint32_t frame_index) override;

	virtual void End(RHICommand *cmd_buffer) override;

  private:
	std::vector<cudaEvent_t> m_start_events;
	std::vector<cudaEvent_t> m_end_events;
};
}        // namespace Ilum::CUDA