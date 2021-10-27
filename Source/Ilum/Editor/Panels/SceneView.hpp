#pragma once

#include "Utils/PCH.hpp"

#include "Editor/Panel.hpp"

namespace Ilum::panel
{
class SceneView : public Panel
{
  public:
	SceneView();

	~SceneView() = default;

	virtual void draw() override;

  private:
	void onResize(VkExtent2D extent);
};
}        // namespace Ilum::panel