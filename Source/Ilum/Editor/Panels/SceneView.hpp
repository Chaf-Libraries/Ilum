#pragma once

#include "Utils/PCH.hpp"

#include "Editor/Panel.hpp"

#include <Graphics/Resource/Image.hpp>

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
	enum class GizmoComponent
	{
		Grid,
		Light,
		AABB,
	};

  private:
	bool                        m_cursor_hidden = false;
	std::pair<int32_t, int32_t> m_last_position;

	float m_camera_speed       = 5.f;
	float m_camera_sensitity = 0.5f;

	uint32_t m_guizmo_operation = 0;

	std::unordered_map<std::string, bool> m_gizmo;

	std::unordered_map<std::string, std::unique_ptr<Graphics::Image>> m_icons;

	std::string m_display_attachment = "";
};
}        // namespace Ilum::panel