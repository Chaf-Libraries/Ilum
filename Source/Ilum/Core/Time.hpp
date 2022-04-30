#pragma once

#include "Singleton.hpp"

#include <chrono>

namespace Ilum
{
class Timer : public Singleton<Timer>
{
  public:
	Timer();
	~Timer() = default;

	float TotalTime();
	float DeltaTime();
	float DeltaTimeSmoothed();
	float FrameRate();
	void Tick();

  private:
	std::chrono::high_resolution_clock::time_point m_start;
	std::chrono::high_resolution_clock::time_point m_tick_start;
	std::chrono::high_resolution_clock::time_point m_tick_end;

	float   m_time                = 0.f;
	float   m_delta_time          = 0.f;
	float   m_delta_time_smoothed = 0.f;
	float   m_accumlate_time      = 0.f;
	float    m_frame_rate          = 0.f;
	uint32_t m_frame_count         = 0u;

	const uint32_t m_accumulate = 5u;
	const float   m_duration   = 500.f;
};
}        // namespace Ilum