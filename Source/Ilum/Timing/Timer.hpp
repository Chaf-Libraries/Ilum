#pragma once

#include "Utils/PCH.hpp"

#include "Engine/Subsystem.hpp"

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

	double   m_time                = 0.0;
	double   m_delta_time          = 0.0;
	double   m_delta_time_smoothed = 0.0;
	double   m_accumlate_time      = 0.0;
	double   m_fps                 = 0.0;
	uint32_t m_frame_count         = 0u;

	const uint32_t m_accumulate = 5u;
	const double   m_duration   = 500.0;
};
}        // namespace Ilum