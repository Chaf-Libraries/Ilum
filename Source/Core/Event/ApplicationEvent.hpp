#pragma once

#include "Event.hpp"

namespace Ilum::Core
{
class WindowClosedEvent : public Event
{
  public:
	WindowClosedEvent() = default;

	static EventType GetType();

	virtual EventType GetEventType() const override;

	virtual const std::string GetName() const override;

	virtual EventCategory GetCategoryFlags() const override;
};

class WindowResizedEvent : public Event
{
  public:
	WindowResizedEvent(uint32_t width, uint32_t height);

	uint32_t GetWidth() const;

	uint32_t GetHeight() const;

	static EventType GetType();

	virtual EventType GetEventType() const override;

	virtual const std::string GetName() const override;

	virtual EventCategory GetCategoryFlags() const override;

	virtual std::string ToString() const override;

  private:
	uint32_t m_width;
	uint32_t m_height;
};

class AppUpdateEvent : public Event
{
  public:
	AppUpdateEvent() = default;

	static EventType GetType();

	virtual EventType GetEventType() const override;

	virtual const std::string GetName() const override;

	virtual EventCategory GetCategoryFlags() const override;
};

class AppRenderEvent : public Event
{
  public:
	AppRenderEvent() = default;

	static EventType  GetType();

	virtual EventType GetEventType() const override;

	virtual const std::string GetName() const override;

	virtual EventCategory GetCategoryFlags() const override;
};
}        // namespace Ilum::Core