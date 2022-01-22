#include "Timer.hpp"

namespace Ilum::Core
{
Timer::Timer()
{
	m_start = std::chrono::high_resolution_clock::now();
}

float Timer::GetFPS()
{
	return m_fps;
}

float Timer::GetTime()
{
	return m_time;
}

float Timer::GetDeltaTime(bool smoothed)
{
	return m_delta_time;
}

void Timer::OnUpdate()
{
	m_update_start                                 = std::chrono::high_resolution_clock::now();
	std::chrono::duration<float, std::milli> delta = m_update_start - m_update_end;

	m_delta_time = static_cast<float>(delta.count());
	m_time       = std::chrono::duration<float, std::milli>(m_update_start - m_start).count();
	m_update_end = std::chrono::high_resolution_clock::now();

	m_delta_time_smoothed = m_delta_time_smoothed * (1.f - 1.f / m_accumulate) + m_delta_time / m_accumulate;

	if (m_accumulate_time < m_duration)
	{
		m_accumulate_time += m_delta_time;
		m_counter++;
	}
	else
	{
		m_fps             = static_cast<float>(m_counter) / (m_accumulate_time / 1000.f);
		m_accumulate_time = 0.f;
		m_counter         = 0;
	}
}

void Timer::Tick()
{
	m_tick = std::chrono::high_resolution_clock::now();
}

float Timer::Elapsed()
{
	return std::chrono::duration<float, std::milli>(std::chrono::high_resolution_clock::now() - m_tick).count();
}
}        // namespace Ilum::Core