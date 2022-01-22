#pragma once

#include <chrono>

namespace Ilum::Core
{
class Timer
{
  public:
	Timer();

	~Timer() = default;

	float GetFPS();

	// Current time in millisecond
	float GetTime();

	// Delta time in millisecond
	float GetDeltaTime(bool smoothed = false);

	// For global timer, call it every frame
	void OnUpdate();

	// Stopwatch tick
	void Tick();

	// Time since last tick in millisecond
	float Elapsed();

  private:
	std::chrono::high_resolution_clock::time_point m_start;
	std::chrono::high_resolution_clock::time_point m_update_start;
	std::chrono::high_resolution_clock::time_point m_update_end;
	std::chrono::high_resolution_clock::time_point m_tick;

	float m_time                = 0.f;
	float m_delta_time          = 0.f;
	float m_delta_time_smoothed = 0.f;
	float m_accumulate_time     = 0.f;
	float m_fps                 = 0.f;

	uint32_t m_counter = 0;

	const uint32_t m_accumulate = 5;
	const float   m_duration   = 500.f;
};
}        // namespace Ilum::Core