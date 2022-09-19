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

  private:
	void DisplayPresent();
	void UpdateCamera();
	void MoveEntity();

  private:
	ViewInfo m_view_info = {};

	struct
	{
		float fov       = 45.f;
		float aspect = 1.f;
		float near_plane = 0.01f;
		float far_plane  = 1000.f;

		float speed     = 1.f;
		float sensitity = 1.f;

		float yaw = 0.f;
		float pitch = 0.f;

		glm::vec2 viewport = glm::vec2(0.f);

		glm::vec3 velocity      = glm::vec3(0.f);
		glm::vec3 position      = glm::vec3(0.f);
		glm::vec3 last_position = glm::vec3(0.f);

		glm::mat4 project_matrix = glm::mat4(1.f);
	} m_camera;

	glm::vec2 m_cursor_position = glm::vec2(0.f);

	bool m_hide_cursor = false;
};
}        // namespace Ilum