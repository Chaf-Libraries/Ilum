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

	virtual void draw(float delta_time) override;

  private:
	void updateMainCamera(float delta_time);

  private:
	void onResize(VkExtent2D extent);

	bool m_cursor_hidden = false;
	std::pair<int32_t, int32_t> m_last_position;
};
}        // namespace Ilum::panel