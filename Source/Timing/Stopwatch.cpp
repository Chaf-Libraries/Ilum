#include "Stopwatch.hpp"

namespace Ilum
{
Stopwatch::Stopwatch()
{
	start();
}

void Stopwatch::start()
{
	m_start = std::chrono::high_resolution_clock::now();
}

float Stopwatch::elapsedSecond() const
{
	return static_cast<float>(std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - m_start).count() / 1000.f);
}

float Stopwatch::elapsedMillisecond() const
{
	return static_cast<float>(std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - m_start).count());
}
}