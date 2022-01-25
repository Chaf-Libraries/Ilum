#pragma once

#include "Event/Event.hpp"
#include "Event/KeyEvent.hpp"
#include "Event/MouseEvent.hpp"
#include "KeyCode.hpp"

#define MAX_KEYS 500
#define MAX_BUTTONS 32

namespace Ilum::Core
{
enum class MouseMode
{
	Visible,
	Hidden,
	Captured
};

class Window;

class Input
{
  public:
	// Key Input
	static bool IsKeyPressed(Key key);
	static bool IsKeyHeld(Key key);

	// Mouse Input
	static bool IsButtonDown(Mouse mouse);
	static void SetMousePosition(float x, float y);
	static void GetMousePosition(float &x, float &y);
	static void GetMouseScrolled(float &x, float &y);
	static void SetMouseMode(MouseMode mode);

	static MouseMode GetMouseMode();

	static void OnEvent(const Event &event);

	static void Flush();

  private:
	bool OnKeyPressed(const KeyPressedEvent &event);
	bool OnKeyReleased(const KeyReleasedEvent &event);
	bool OnMouseButtonPressed(const MouseButtonPressedEvent &event);
	bool OnMouseButtonReleased(const MouseButtonReleasedEvent &event);
	bool OnMouseMoved(const MouseMovedEvent &event);
	bool OnMouseScrolled(const MouseScrolledEvent &event);

  private:
	static Input &GetInstance();

	void OnEvent_(const Event &event);

	bool m_key_pressed[MAX_KEYS] = {false};
	bool m_key_held[MAX_KEYS]    = {false};

	bool m_button[MAX_BUTTONS] = {false};

	MouseMode m_mouse_mode        = MouseMode::Visible;
	float     m_mouse_position[2] = {0.f, 0.f};
	float     m_mouse_scrolled[2] = {0.f, 0.f};
};
}        // namespace Ilum::Core