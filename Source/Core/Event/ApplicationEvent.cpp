#include "ApplicationEvent.hpp"

#include <sstream>

namespace Ilum::Core
{
EventType WindowClosedEvent::GetType()
{
	return EventType::WindowClose;
}

EventType WindowClosedEvent::GetEventType() const
{
	return EventType::WindowClose;
}

const std::string WindowClosedEvent::GetName() const
{
	return "WindowClosedEvent";
}

EventCategory WindowClosedEvent::GetCategoryFlags() const
{
	return EventCategory::Application;
}

WindowResizedEvent::WindowResizedEvent(uint32_t width, uint32_t height):
    m_width(width), m_height(height)
{
}

uint32_t WindowResizedEvent::GetWidth() const
{
	return m_width;
}

uint32_t WindowResizedEvent::GetHeight() const
{
	return m_height;
}

EventType WindowResizedEvent::GetType()
{
	return EventType::WindowResize;
}

EventType WindowResizedEvent::GetEventType() const
{
	return EventType::WindowResize;
}

const std::string WindowResizedEvent::GetName() const
{
	return "WindowResizedEvent";
}

EventCategory WindowResizedEvent::GetCategoryFlags() const
{
	return EventCategory::Application;
}

std::string WindowResizedEvent::ToString() const
{
	std::stringstream ss;
	ss << "WindowResizedEvent: " << m_width << ", " << m_height;
	return ss.str();
}

EventType AppUpdateEvent::GetType()
{
	return EventType::AppUpdate;
}

EventType AppUpdateEvent::GetEventType() const
{
	return EventType::AppUpdate;
}

const std::string AppUpdateEvent::GetName() const
{
	return "AppUpdateEvent";
}

EventCategory AppUpdateEvent::GetCategoryFlags() const
{
	return EventCategory::Application;
}

EventType AppRenderEvent::GetType()
{
	return EventType::AppRender;
}

EventType AppRenderEvent::GetEventType() const
{
	return EventType::AppRender;
}

const std::string AppRenderEvent::GetName() const
{
	return "AppRenderEvent";
}

EventCategory AppRenderEvent::GetCategoryFlags() const
{
	return EventCategory::Application;
}
}        // namespace Ilum::Core