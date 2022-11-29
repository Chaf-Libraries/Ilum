#pragma once

#include "Precompile.hpp"
#include "Delegates.hpp"
#include "Input.hpp"

struct GLFWwindow;

namespace Ilum
{
class EXPORT_API Window
{
  public:
	Window(const std::string &title, const std::string &icon, uint32_t width = 0, uint32_t height = 0);

	~Window();

	bool Tick();

	bool IsKeyDown(int32_t key) const;

	bool IsMouseButtonDown(int32_t button) const;

	void SetTitle(const std::string &title);

	GLFWwindow *GetHandle() const;

	void *GetNativeHandle() const;

	uint32_t GetWidth() const;

	uint32_t GetHeight() const;

	bool IsKeyPressed(KeyCode keycode);

	bool IsMouseButtonPressed(MouseCode button);

	glm::vec2 GetMousePosition();

	void SetCursorPosition(const glm::vec2 &pos);

  public:
	MulticastDelegate<>                                   OnResetFunc;
	MulticastDelegate<int32_t, int32_t, int32_t, int32_t> OnKeyFunc;
	MulticastDelegate<uint32_t>                           OnCharFunc;
	MulticastDelegate<int32_t, uint32_t>                  OnCharModsFunc;
	MulticastDelegate<int32_t, int32_t, int32_t>          OnMouseButtonFunc;
	MulticastDelegate<double, double>                     OnCursorPosFunc;
	MulticastDelegate<int32_t>                            OnCursorEnterFunc;
	MulticastDelegate<double, double>                     OnScrollFunc;
	MulticastDelegate<int32_t, const char **>             OnDropFunc;
	MulticastDelegate<int32_t, int32_t>                   OnWindowSizeFunc;
	MulticastDelegate<>                                   OnWindowCloseFunc;

  private:
	GLFWwindow *m_handle = nullptr;

	uint32_t    m_width;
	uint32_t    m_height;
	std::string m_title;
	float       m_mouse_wheel_h = 0.0f;
	float       m_mouse_wheel   = 0.0f;
	float       m_pos_delta_x   = 0.f;
	float       m_pos_delta_y   = 0.f;
	float       m_pos_last_x    = 0.f;
	float       m_pos_last_y    = 0.f;
};
}        // namespace Ilum