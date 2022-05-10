#pragma once

#include "Delegates.hpp"

#include <GLFW/glfw3.h>

#include <string>

namespace Ilum
{
class Window
{
  public:
	Window(const std::string &title, const std::string &icon, uint32_t width, uint32_t height);
	~Window();

	bool Tick();

	bool IsKeyDown(int32_t key) const;
	bool IsMouseButtonDown(int32_t button) const;

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

  public:
	GLFWwindow *m_handle = nullptr;

	uint32_t    m_width;
	uint32_t    m_height;
	std::string m_title;
	float       m_mouse_wheel_h = 0.0f;
	float       m_mouse_wheel   = 0.0f;
};
}        // namespace Ilum