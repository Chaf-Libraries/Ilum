#include "Time.hpp"

#include <chrono>

namespace Ilum
{
struct Timer::Impl
{
	Timer::Impl(std::chrono::high_resolution_clock::time_point start, std::chrono::high_resolution_clock::time_point end):
	    start(start), tick_end(end)
	{
	}

	std::chrono::high_resolution_clock::time_point start;
	std::chrono::high_resolution_clock::time_point tick_start;
	std::chrono::high_resolution_clock::time_point tick_end;

	float    time                = 0.f;
	float    delta_time          = 0.f;
	float    delta_time_smoothed = 0.f;
	float    accumlate_time      = 0.f;
	float    frame_rate          = 0.f;
	uint32_t frame_count         = 0u;

	const uint32_t accumulate = 5u;
	const float    duration   = 500.f;
};

Timer::Timer()
{
	p_impl = new Impl(std::chrono::high_resolution_clock::now(), std::chrono::high_resolution_clock::now());
}

Timer::~Timer()
{
	delete p_impl;
	p_impl = nullptr;
}

Timer &Timer::GetInstance()
{
	static Timer timer;
	return timer;
}

float Timer::TotalTime()
{
	return p_impl->time;
}

float Timer::DeltaTime()
{
	return p_impl->delta_time;
}

float Timer::DeltaTimeSmoothed()
{
	return p_impl->delta_time_smoothed;
}

float Timer::FrameRate()
{
	return p_impl->frame_rate;
}

void Timer::Tick()
{
	p_impl->tick_start                                    = std::chrono::high_resolution_clock::now();
	std::chrono::duration<float, std::milli> delta = p_impl->tick_start - p_impl->tick_end;

	p_impl->delta_time = delta.count();
	p_impl->time       = std::chrono::duration<float, std::milli>(p_impl->tick_start - p_impl->start).count();
	p_impl->tick_end   = std::chrono::high_resolution_clock::now();

	p_impl->delta_time_smoothed = p_impl->delta_time_smoothed * (1.f - 1.f / p_impl->accumulate) + p_impl->delta_time / p_impl->accumulate;

	if (p_impl->accumlate_time < p_impl->duration)
	{
		p_impl->accumlate_time += p_impl->delta_time;
		p_impl->frame_count++;
	}
	else
	{
		p_impl->frame_rate            = static_cast<float>(p_impl->frame_count) / (p_impl->accumlate_time / 1000.f);
		p_impl->accumlate_time = 0.f;
		p_impl->frame_count    = 0;
	}
}
}        // namespace Ilum