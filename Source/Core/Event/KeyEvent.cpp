#include "KeyEvent.hpp"

#include <sstream>

namespace Ilum::Core
{
uint32_t KeyEvent::GetKeyCode() const
{
	return m_keycode;
}

EventCategory KeyEvent::GetCategoryFlags() const
{
	return EventCategory::Input | EventCategory::Keyboard;
}

KeyEvent::KeyEvent(uint32_t keycode) :
    m_keycode(keycode)
{
}

KeyPressedEvent::KeyPressedEvent(uint32_t keycode, uint32_t repeat_count) :
    KeyEvent(keycode), m_repeat_count(repeat_count)
{
}

uint32_t KeyPressedEvent::GetRepeatCount() const
{
	return m_repeat_count;
}

EventType KeyPressedEvent::GetType()
{
	return EventType::KeyPressed;
}

EventType KeyPressedEvent::GetEventType() const
{
	return EventType::KeyPressed;
}

const std::string KeyPressedEvent::GetName() const
{
	return "KeyPressEvent";
}

std::string KeyPressedEvent::ToString() const
{
	std::stringstream ss;
	ss << "KeyPressEvent: " << m_keycode << "(" << m_repeat_count << " repeats)";
	return ss.str();
}

KeyReleasedEvent::KeyReleasedEvent(uint32_t keycode):
    KeyEvent(keycode)
{

}

EventType KeyReleasedEvent::GetType()
{
	return EventType::KeyReleased;
}

EventType KeyReleasedEvent::GetEventType() const
{
	return EventType::KeyReleased;
}

const std::string KeyReleasedEvent::GetName() const
{
	return "KeyReleasedEvent";
}

std::string KeyReleasedEvent::ToString() const
{
	std::stringstream ss;
	ss << "KeyReleasedEvent: " << m_keycode;
	return ss.str();
}

KeyTypedEvent::KeyTypedEvent(uint32_t keycode):
    KeyEvent(keycode)
{
}

EventType KeyTypedEvent::GetType()
{
	return EventType::KeyTyped;
}

EventType KeyTypedEvent::GetEventType() const
{
	return EventType::KeyTyped;
}

const std::string KeyTypedEvent::GetName() const
{
	return "KeyTypedEvent";
}

std::string KeyTypedEvent::ToString() const
{
	std::stringstream ss;
	ss << "KeyTypedEvent: " << m_keycode;
	return ss.str();
}
}        // namespace Ilum::Core