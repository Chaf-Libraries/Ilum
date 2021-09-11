#pragma once

#include "Core/Engine/PCH.hpp"

#include "Core/Engine/Subsystem.hpp"

namespace Ilum
{
class Timer : public TSubsystem<Timer>
{
  public:
	Timer(Context *context = nullptr);

	~Timer() = default;

	void onTick(float delta_time) override;

	double getFPS() const;

	double getTimeMillisecond() const;

	double getTimeSecond() const;

	double getDeltaTimeMillisecond() const;

	double getDeltaTimeSecond() const;

	double getDeltaTimeMillisecondSmoothed() const;

	double getDeltaTimeSecondSmoothed() const;

  private:
	std::chrono::high_resolution_clock::time_point m_start;
	std::chrono::high_resolution_clock::time_point m_tick_start;
	std::chrono::high_resolution_clock::time_point m_tick_end;

	double   m_time                = 0.f;
	double   m_delta_time          = 0.f;
	double   m_delta_time_smoothed = 0.f;
	double   m_accumlate_time      = 0.f;
	double   m_fps                 = 0.f;
	uint32_t m_frame_count         = 0;

	const uint32_t m_accumulate = 5;
	const double   m_duration   = 500.f;
};
}        // namespace Ilum