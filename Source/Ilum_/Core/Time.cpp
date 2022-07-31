#include "Time.hpp"

namespace Ilum
{
Timer::Timer():
    m_start(std::chrono::high_resolution_clock::now()),
    m_tick_end(std::chrono::high_resolution_clock::now())
{
}

float Timer::TotalTime()
{
	return m_time;
}

float Timer::DeltaTime()
{
	return m_delta_time;
}

float Timer::DeltaTimeSmoothed()
{
	return m_delta_time_smoothed;
}

float Timer::FrameRate()
{
	return m_frame_rate;
}

void Timer::Tick()
{
	m_tick_start                                    = std::chrono::high_resolution_clock::now();
	std::chrono::duration<float, std::milli> delta = m_tick_start - m_tick_end;

	m_delta_time = delta.count();
	m_time       = std::chrono::duration<float, std::milli>(m_tick_start - m_start).count();
	m_tick_end   = std::chrono::high_resolution_clock::now();

	m_delta_time_smoothed = m_delta_time_smoothed * (1.f - 1.f / m_accumulate) + m_delta_time / m_accumulate;

	if (m_accumlate_time < m_duration)
	{
		m_accumlate_time += m_delta_time;
		m_frame_count++;
	}
	else
	{
		m_frame_rate            = static_cast<float>(m_frame_count) / (m_accumlate_time / 1000.f);
		m_accumlate_time = 0.f;
		m_frame_count    = 0;
	}
}
}        // namespace Ilum