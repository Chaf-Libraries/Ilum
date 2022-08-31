#pragma once

#include "Widget.hpp"

namespace Ilum
{
class SceneView : public Widget
{
  public:
	SceneView(Editor *editor);

	~SceneView();

	virtual void Tick() override;
};
}        // namespace Ilum