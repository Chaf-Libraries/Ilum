#pragma once

#include "Utils/PCH.hpp"

#include "Editor/Panel.hpp"

#include "Timing/Stopwatch.hpp"

namespace Ilum::panel
{
class RendererInspector : public Panel
{
  public:
	RendererInspector();

	~RendererInspector() = default;

	virtual void draw(float delta_time) override;

  private:
	Stopwatch m_stopwatch;
	std::vector<float> m_frame_times;
};
}        // namespace Ilum::panel