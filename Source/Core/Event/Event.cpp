#include "Event.hpp"

namespace Ilum::Core
{
bool Event::isInCategory(EventCategory category)
{
	return static_cast<uint32_t>(GetCategoryFlags() & category);
}

std::string Event::ToString() const
{
	return GetName();
}

bool Event::operator()()
{
	return m_handle;
}

EventDispatcher::EventDispatcher(const Event &event):
    m_event(event)
{
}
}        // namespace Ilum::Core