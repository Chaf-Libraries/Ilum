#pragma once

#include "Widget.hpp"

#include <vector>

namespace Ilum
{
class RenderGraphInspector : public Widget
{
  public:
	RenderGraphInspector(Editor *editor);

	~RenderGraphInspector();

	virtual void Tick() override;

  private:
	std::vector<float> m_frame_times;
	
};
}        // namespace Ilum