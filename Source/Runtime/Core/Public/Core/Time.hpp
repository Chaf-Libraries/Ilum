#pragma once

#include "Precompile.hpp"

namespace Ilum
{
class  Timer
{
  public:
	Timer();

	~Timer();

	static Timer &GetInstance();

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