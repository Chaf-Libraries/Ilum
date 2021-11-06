#pragma once

#include "Utils/PCH.hpp"

#include "Editor/Panel.hpp"

#include "Graphics/Image/Image.hpp"

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

	void onResize(VkExtent2D extent);

	void showToolBar();

  private:
	bool                        m_cursor_hidden = false;
	std::pair<int32_t, int32_t> m_last_position;

	uint32_t m_guizmo_operation = 0;
	bool     m_grid             = true;

	std::unordered_map<std::string, Image> m_icons;
};
}        // namespace Ilum::panel