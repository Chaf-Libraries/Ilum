#pragma once

#include "Event.h"

namespace Ilum
{
template <typename... Args>
inline uint32_t Event<Args...>::subscribe(CallbackFunc callback)
{
	uint32_t id = m_avaliable_id++;
	m_subscribers.emplace(id, callback);
	return id;
}

template <typename... Args>
inline uint32_t Event<Args...>::operator+=(CallbackFunc callback)
{
	return subscribe(callback);
}

template <typename... Args>
inline bool Event<Args...>::unsubscribe(uint32_t subscriber)
{
	return m_subscribers.erase(subscriber);
}

template <typename... Args>
inline bool Event<Args...>::operator-=(uint32_t subscriber)
{
	return unsubscribe(subscriber);
}

template <typename... Args>
inline void Event<Args...>::clear()
{
	m_subscribers.clear();
}

template <typename... Args>
inline uint32_t Event<Args...>::getSubscribersCount() const
{
	return m_subscribers.size();
}

template <typename... Args>
inline void Event<Args...>::invoke(Args... args)
{
	for (auto const &[key, val] : m_subscribers)
	{
		val(args...);
	}
}
}