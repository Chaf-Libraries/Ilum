#pragma once

#include "Widget.hpp"

namespace Ilum
{
class SceneInspector : public Widget
{
  public:
	SceneInspector(Editor *editor);

	~SceneInspector();

	virtual void Tick() override;
};
}        // namespace Ilum