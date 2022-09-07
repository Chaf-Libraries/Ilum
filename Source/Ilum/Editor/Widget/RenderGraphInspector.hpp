#pragma once

#include "Widget.hpp"

namespace Ilum
{
class RenderGraphInspector : public Widget
{
  public:
	RenderGraphInspector(Editor *editor);

	~RenderGraphInspector();

	virtual void Tick() override;
};
}        // namespace Ilum