#pragma once

#include "Widget.hpp"

#include <MaterialGraph/MaterialGraph.hpp>
#include <MaterialGraph/MaterialNode.hpp>

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

	std::string m_material_name = "";

	size_t m_uuid = size_t(~0);

	size_t m_current_handle = 0;
};
}        // namespace Ilum