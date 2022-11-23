#pragma once

namespace Ilum
{
class __declspec(dllexport) Timer
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