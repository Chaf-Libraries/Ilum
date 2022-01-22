#include "Input.hpp"
#include "Window.hpp"

namespace Ilum::Core
{
bool Input::IsKeyPressed(Key key)
{
	return GetInstance().m_key_pressed[(uint32_t) key];
}

bool Input::IsKeyHeld(Key key)
{
	return GetInstance().m_key_held[(uint32_t) key];
}

bool Input::IsButtonDown(Mouse mouse)
{
	return GetInstance().m_button[(uint32_t) mouse];
}

void Input::SetMousePosition(float x, float y)
{
	if (Window::GetInstance())
	{
		Window::GetInstance()->SetMousePosition(x, y);
	}
}

void Input::GetMousePosition(float &x, float &y)
{
	x = GetInstance().m_mouse_position[0];
	y = GetInstance().m_mouse_position[1];
}

void Input::GetMouseScrolled(float &x, float &y)
{
	x = GetInstance().m_mouse_scrolled[0];
	y = GetInstance().m_mouse_scrolled[1];
}

void Input::SetMouseMode(MouseMode mode)
{
	GetInstance().m_mouse_mode = mode;

	if (Window::GetInstance())
	{
		Window::GetInstance()->HideMouse(mode == MouseMode::Hidden);
	}
}

MouseMode Input::GetMouseMode()
{
	return GetInstance().m_mouse_mode;
}

void Input::OnEvent(Event &event)
{
	GetInstance().OnEvent_(event);
}

void Input::Flush()
{
	memset(GetInstance().m_key_pressed, 0, MAX_KEYS);
	memset(GetInstance().m_button, 0, MAX_BUTTONS);

	GetInstance().m_mouse_scrolled[0] = 0.f;
	GetInstance().m_mouse_scrolled[1] = 0.f;
}

bool Input::OnKeyPressed(KeyPressedEvent &event)
{
	m_key_pressed[uint32_t(event.GetKeyCode())] = event.GetRepeatCount() < 1;
	m_key_held[uint32_t(event.GetKeyCode())]    = true;
	return false;
}

bool Input::OnKeyReleased(KeyReleasedEvent &event)
{
	m_key_pressed[uint32_t(event.GetKeyCode())] = false;
	m_key_held[uint32_t(event.GetKeyCode())]    = false;
	return false;
}

bool Input::OnMouseButtonPressed(MouseButtonPressedEvent &event)
{
	m_button[uint32_t(event.GetMouseButton())] = true;
	return false;
}

bool Input::OnMouseButtonReleased(MouseButtonReleasedEvent &event)
{
	m_button[uint32_t(event.GetMouseButton())] = false;
	return false;
}

bool Input::OnMouseMoved(MouseMovedEvent &event)
{
	m_mouse_position[0] = event.GetX();
	m_mouse_position[1] = event.GetY();
	return false;
}

bool Input::OnMouseScrolled(MouseScrolledEvent &event)
{
	m_mouse_scrolled[0] = event.GetOffsetX();
	m_mouse_scrolled[1] = event.GetOffsetY();
	return false;
}

Input &Input::GetInstance()
{
	static Input input;
	return input;
}

void Input::OnEvent_(Event &event)
{
	EventDispatcher dispatcher(event);
	dispatcher.Dispatch<KeyPressedEvent>(std::bind(&Input::OnKeyPressed, this, std::placeholders::_1));
	dispatcher.Dispatch<KeyReleasedEvent>(std::bind(&Input::OnKeyReleased, this, std::placeholders::_1));
	dispatcher.Dispatch<MouseButtonPressedEvent>(std::bind(&Input::OnMouseButtonPressed, this, std::placeholders::_1));
	dispatcher.Dispatch<MouseButtonReleasedEvent>(std::bind(&Input::OnMouseButtonReleased, this, std::placeholders::_1));
	dispatcher.Dispatch<MouseMovedEvent>(std::bind(&Input::OnMouseMoved, this, std::placeholders::_1));
	dispatcher.Dispatch<MouseScrolledEvent>(std::bind(&Input::OnMouseScrolled, this, std::placeholders::_1));
}
}        // namespace Ilum::Core