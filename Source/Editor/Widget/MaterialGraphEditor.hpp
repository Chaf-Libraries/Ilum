#pragma once

#include "Widget.hpp"

#include <RenderCore/MaterialGraph/MaterialGraph.hpp>
#include <RenderCore/MaterialGraph/MaterialNode.hpp>

struct ImNodesEditorContext;

namespace Ilum
{
class MaterialGraphEditor : public Widget
{
  public:
	MaterialGraphEditor(Editor *editor);

	~MaterialGraphEditor();

	virtual void Tick() override;

  private:
	void DrawMenu();

  private:
	ImNodesEditorContext *m_context = nullptr;

	MaterialGraphDesc m_desc;

	size_t        m_current_handle = 0;
};
}        // namespace Ilum