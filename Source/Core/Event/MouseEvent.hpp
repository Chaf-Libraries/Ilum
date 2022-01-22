#pragma once

#include "Event.hpp"

namespace Ilum::Core
{
class MouseMovedEvent : public Event
{
  public:
	MouseMovedEvent(float x, float y);

	float GetX() const;

	float GetY() const;

	static EventType GetType();

	virtual EventType GetEventType() const override;

	virtual const std::string GetName() const override;

	virtual EventCategory GetCategoryFlags() const override;

	virtual std::string ToString() const override;

  private:
	float m_mouse_x, m_mouse_y;
};

class MouseScrolledEvent : public Event
{
  public:
	MouseScrolledEvent(float offset_x, float offset_y);

	float GetOffsetX() const;

	float GetOffsetY() const;

	static EventType GetType();

	virtual EventType GetEventType() const override;

	virtual const std::string GetName() const override;

	virtual EventCategory GetCategoryFlags() const override;

	virtual std::string ToString() const override;

  private:
	float m_offset_x;
	float m_offset_y;
};

class MouseButtonEvent : public Event
{
  public:
	uint32_t GetMouseButton() const;

	virtual EventCategory GetCategoryFlags() const override;

  protected:
	MouseButtonEvent(uint32_t button);

	uint32_t m_button;
};

class MouseButtonPressedEvent : public MouseButtonEvent
{
  public:
	MouseButtonPressedEvent(uint32_t button);

	static EventType GetType();

	virtual EventType GetEventType() const override;

	virtual const std::string GetName() const override;

	virtual std::string ToString() const override;
};

class MouseButtonReleasedEvent : public MouseButtonEvent
{
  public:
	MouseButtonReleasedEvent(uint32_t button);

	static EventType GetType();

	virtual EventType GetEventType() const override;

	virtual const std::string GetName() const override;

	virtual std::string ToString() const override;
};
}        // namespace Ilum::Core