#pragma once

#include "Widget.hpp"

#include <RenderCore/RenderGraph/RenderGraph.hpp>

namespace Ilum
{
class RenderGraphEditor : public Widget
{
  public:
	RenderGraphEditor(Editor* editor);

	~RenderGraphEditor();

	virtual void Tick() override;

  private:
	RenderGraphDesc m_desc;
	size_t          m_current_handle = 0;
};
}        // namespace Ilum