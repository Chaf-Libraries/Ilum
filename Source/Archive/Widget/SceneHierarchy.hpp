#pragma once

#include "Widget.hpp"

namespace Ilum
{
class Entity;

class SceneHierarchy : public Widget
{
  public:
	SceneHierarchy(Editor *editor);

	~SceneHierarchy();

	virtual void Tick() override;

  private:
	void DrawNode(Entity &entity);
};
}        // namespace Ilum