#pragma once

#include "Widget.hpp"

namespace Ilum
{
class MaterialGraphEditor : public Widget
{
  public:
	MaterialGraphEditor(Editor *editor);

	~MaterialGraphEditor();

	virtual void Tick() override;
};
}        // namespace Ilum