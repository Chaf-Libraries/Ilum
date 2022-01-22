#pragma once

#include "Event.hpp"

namespace Ilum::Core
{
class KeyEvent : public Event
{
  public:
	uint32_t GetKeyCode() const;

	virtual EventCategory GetCategoryFlags() const override;

  protected:
	KeyEvent(uint32_t keycode);

	uint32_t m_keycode;
};

class KeyPressedEvent : public KeyEvent
{
  public:
	KeyPressedEvent(uint32_t keycode, uint32_t repeat_count);

	uint32_t GetRepeatCount() const;

		static EventType GetType();

	virtual EventType GetEventType() const override;

	virtual const std::string GetName() const override;

	virtual std::string ToString() const override;

  private:
	uint32_t m_repeat_count;
};

class KeyReleasedEvent : public KeyEvent
{
  public:
	KeyReleasedEvent(uint32_t keycode);

	static EventType GetType();

	virtual EventType GetEventType() const override;

	virtual const std::string GetName() const override;

	virtual std::string ToString() const override;
};

class KeyTypedEvent : public KeyEvent
{
  public:
	KeyTypedEvent(uint32_t keycode);

	static EventType GetType();

	virtual EventType GetEventType() const override;

	virtual const std::string GetName() const override;

	virtual std::string ToString() const override;
};
}        // namespace Ilum::Core