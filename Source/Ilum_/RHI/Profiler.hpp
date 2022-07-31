#pragma once

#include <volk.h>

#include <chrono>
#include <thread>

namespace Ilum
{
class RHIDevice;
class CommandBuffer;

struct ProfileState
{
	std::chrono::time_point<std::chrono::high_resolution_clock> cpu_start;
	std::chrono::time_point<std::chrono::high_resolution_clock> cpu_end;

	uint64_t gpu_start = 0;
	uint64_t gpu_end   = 0;

	float cpu_time = 0.f;
	float gpu_time = 0.f;

	std::thread::id thread_id;
};

class Profiler
{
  public:
	Profiler(RHIDevice *device);
	~Profiler();

	const ProfileState &GetProfileState() const;

	void Begin(CommandBuffer& cmd_buffer);

	void End(CommandBuffer &cmd_buffer);

  private:
	RHIDevice *p_device = nullptr;

	ProfileState m_state;
	std::vector<VkQueryPool> m_query_pools = {};
};
}        // namespace Ilum