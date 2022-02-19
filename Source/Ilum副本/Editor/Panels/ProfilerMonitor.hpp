#pragma once

#include "Utils/PCH.hpp"

#include "Editor/Panel.hpp"

#include "Timing/Stopwatch.hpp"

namespace Ilum::panel
{
class ProfilerMonitor : public Panel
{
  public:
	ProfilerMonitor();

	~ProfilerMonitor() = default;

	virtual void draw(float delta_time) override;

  private:
	Stopwatch m_stopwatch;
	std::map<std::string, std::pair<float, float>> m_profile_result;
	std::vector<float>                                        m_frame_times;
};
}        // namespace Ilum::panel