#pragma once

#include <functional>
#include <ostream>

namespace Ilum::Core
{
enum class EventType : uint32_t
{
	None = 0,

	WindowClose,
	WindowResize,

	AppUpdate,
	AppRender,

	KeyPressed,
	KeyReleased,
	KeyTyped,

	MouseButtonPressed,
	MouseButtonReleased,
	MouseMoved,
	MouseScrolled
};

enum class EventCategory : uint32_t
{
	None        = 0,
	Application = 1 << 0,
	Input       = 1 << 1,
	Keyboard    = 1 << 2,
	Mouse       = 1 << 3,
	MouseButton = 1 << 4,
};

inline EventCategory operator|(EventCategory lhs, EventCategory rhs)
{
	return static_cast<EventCategory>(static_cast<uint32_t>(lhs) | static_cast<uint32_t>(rhs));
}

inline EventCategory operator&(EventCategory lhs, EventCategory rhs)
{
	return static_cast<EventCategory>(static_cast<uint32_t>(lhs) & static_cast<uint32_t>(rhs));
}

class EventDispatcher;

class Event
{
	friend class EventDispatcher;

  public:
	virtual EventType GetEventType() const = 0;

	virtual const std::string GetName() const = 0;

	virtual EventCategory GetCategoryFlags() const = 0;

	bool isInCategory(EventCategory category);

	virtual std::string ToString() const;

	bool operator()();

  private:
	bool m_handle = false;
};

inline std::ostream &operator<<(std::ostream &os, const Event &e)
{
	return os << e.ToString();
}

class EventDispatcher
{
	template <typename T>
	using EventFunc = std::function<bool(T &)>;

  public:
	EventDispatcher(Event &event);

	template <typename T>
	inline bool Dispatch(EventFunc<T> func)
	{
		if (m_event.GetEventType() == T::GetType())
		{
			m_event.m_handle = func(*static_cast<T *>(&m_event));
			return true;
		}
		return false;
	}

  private:
	Event &m_event;
};

}        // namespace Ilum::Core