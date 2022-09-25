#pragma once

#include "Precompile.hpp"
#include "Singleton.hpp"

namespace Ilum
{
class Timer : public Singleton<Timer>
{
  public:
	Timer();

	~Timer();

	float TotalTime();
	float DeltaTime();
	float DeltaTimeSmoothed();
	float FrameRate();
	void  Tick();

  private:
	struct Impl;
	Impl *p_impl = nullptr;
};
}        // namespace Ilum