#pragma once

#include "Utils/PCH.hpp"

#include "Editor/Panel.hpp"

#include <Core/Timer.hpp>

namespace Ilum::panel
{
class ProfilerMonitor : public Panel
{
  public:
	ProfilerMonitor();

	~ProfilerMonitor() = default;

	virtual void draw(float delta_time) override;

  private:
	Core::Timer m_timer;
	std::map<std::string, std::pair<float, float>> m_profile_result;
	std::vector<float>                                        m_frame_times;
};
}        // namespace Ilum::panel