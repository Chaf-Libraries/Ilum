#include "MouseEvent.hpp"

#include <sstream>

namespace Ilum::Core
{
MouseMovedEvent::MouseMovedEvent(float x, float y) :
    m_mouse_x(x), m_mouse_y(y)
{
}

float MouseMovedEvent::GetX() const
{
	return m_mouse_x;
}

float MouseMovedEvent::GetY() const
{
	return m_mouse_y;
}

EventType MouseMovedEvent::GetType()
{
	return EventType::MouseMoved;
}

EventType MouseMovedEvent::GetEventType() const
{
	return EventType::MouseMoved;
}

const std::string MouseMovedEvent::GetName() const
{
	return "MouseMovedEvent";
}

EventCategory MouseMovedEvent::GetCategoryFlags() const
{
	return EventCategory::Input | EventCategory::Mouse;
}

std::string MouseMovedEvent::ToString() const
{
	std::stringstream ss;
	ss << "MouseMovedEvent: " << m_mouse_x << ", " << m_mouse_y;
	return ss.str();
}

MouseScrolledEvent::MouseScrolledEvent(float offset_x, float offset_y) :
    m_offset_x(offset_x), m_offset_y(offset_y)
{
}

float MouseScrolledEvent::GetOffsetX() const
{
	return m_offset_x;
}

float MouseScrolledEvent::GetOffsetY() const
{
	return m_offset_y;
}

EventType MouseScrolledEvent::GetType()
{
	return EventType::MouseScrolled;
}

EventType MouseScrolledEvent::GetEventType() const
{
	return EventType::MouseScrolled;
}

const std::string MouseScrolledEvent::GetName() const
{
	return "MouseScrolledEvent";
}

EventCategory MouseScrolledEvent::GetCategoryFlags() const
{
	return EventCategory::Input | EventCategory::Mouse;
}

std::string MouseScrolledEvent::ToString() const
{
	std::stringstream ss;
	ss << "MouseScrolledEvent: " << m_offset_x << ", " << m_offset_y;
	return ss.str();
}

uint32_t MouseButtonEvent::GetMouseButton() const
{
	return m_button;
}

EventCategory MouseButtonEvent::GetCategoryFlags() const
{
	return EventCategory::Input | EventCategory::MouseButton;
}

MouseButtonEvent::MouseButtonEvent(uint32_t button) :
    m_button(button)
{
}

MouseButtonPressedEvent::MouseButtonPressedEvent(uint32_t button) :
    MouseButtonEvent(button)
{
}

EventType MouseButtonPressedEvent::GetType()
{
	return EventType::MouseButtonPressed;
}

EventType MouseButtonPressedEvent::GetEventType() const
{
	return EventType::MouseButtonPressed;
}

const std::string MouseButtonPressedEvent::GetName() const
{
	return "MouseButtonPressedEvent";
}

std::string MouseButtonPressedEvent::ToString() const
{
	std::stringstream ss;
	ss << "MouseButtonPressedEvent: " << m_button;
	return ss.str();
}

MouseButtonReleasedEvent::MouseButtonReleasedEvent(uint32_t button) :
    MouseButtonEvent(button)
{
}

EventType MouseButtonReleasedEvent::GetType()
{
	return EventType::MouseButtonReleased;
}

EventType MouseButtonReleasedEvent::GetEventType() const
{
	return EventType::MouseButtonReleased;
}

const std::string MouseButtonReleasedEvent::GetName() const
{
	return "MouseButtonReleasedEvent";
}

std::string MouseButtonReleasedEvent::ToString() const
{
	std::stringstream ss;
	ss << "MouseButtonReleasedEvent: " << m_button;
	return ss.str();
}
}        // namespace Ilum::Core