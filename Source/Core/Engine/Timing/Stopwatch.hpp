#pragma once

#include "Core/Engine/PCH.hpp"

namespace Ilum
{
class Stopwatch
{
  public:
	Stopwatch();

	~Stopwatch() = default;

	void start();

	float elapsedSecond() const;

	float elapsedMillisecond() const;

  private:
	std::chrono::time_point<std::chrono::high_resolution_clock> m_start;
};
}