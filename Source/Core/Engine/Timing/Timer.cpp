#include "Timer.hpp"

namespace Ilum
{
Timer::Timer(Context *context) :
    TSubsystem<Timer>(context)
{
	m_start    = std::chrono::high_resolution_clock::now();
	m_tick_end = std::chrono::high_resolution_clock::now();
}

void Timer::onTick(float delta_time)
{
	m_tick_start                                    = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double, std::milli> delta = m_tick_start - m_tick_end;

	m_delta_time = static_cast<double>(delta.count());
	m_time       = static_cast<double>(std::chrono::duration<double, std::milli>(m_tick_start - m_start).count());
	m_tick_end   = std::chrono::high_resolution_clock::now();

	m_delta_time_smoothed = m_delta_time_smoothed * (1 - 1.0 / static_cast<float>(m_accumulate)) + m_delta_time / static_cast<float>(m_accumulate);

	if (m_accumlate_time < m_duration)
	{
		m_accumlate_time += m_delta_time;
		m_frame_count++;
	}
	else
	{
		m_fps            = static_cast<double>(m_frame_count) / (m_accumlate_time / 1000.f);
		m_accumlate_time = 0.f;
		m_frame_count    = 0;
	}
}

double Timer::getFPS() const
{
	return m_fps;
}

double Timer::getTimeMillisecond() const
{
	return m_time;
}

double Timer::getTimeSecond() const
{
	return m_time / 1000.0;
}

double Timer::getDeltaTimeMillisecond() const
{
	return m_delta_time;
}

double Timer::getDeltaTimeSecond() const
{
	return m_delta_time / 1000.0;
}

double Timer::getDeltaTimeMillisecondSmoothed() const
{
	return m_delta_time_smoothed;
}

double Timer::getDeltaTimeSecondSmoothed() const
{
	return m_delta_time_smoothed / 1000.0;
}
}