#pragma once

#include "Event.hpp"

namespace Ilum::Core
{
template <typename... Args>
inline uint32_t Event<Args...>::Subscribe(CallbackFunc callback)
{
	uint32_t id = m_avaliable_id++;
	m_subscribers.emplace(id, callback);
	return id;
}

template <typename... Args>
inline uint32_t Event<Args...>::operator+=(CallbackFunc callback)
{
	return Subscribe(callback);
}

template <typename... Args>
inline bool Event<Args...>::Unsubscribe(uint32_t subscriber)
{
	return m_subscribers.erase(subscriber);
}

template <typename... Args>
inline bool Event<Args...>::operator-=(uint32_t subscriber)
{
	return Unsubscribe(subscriber);
}

template <typename... Args>
inline void Event<Args...>::Clear()
{
	m_subscribers.clear();
}

template <typename... Args>
inline uint32_t Event<Args...>::GetSubscribersCount() const
{
	return m_subscribers.size();
}

template <typename... Args>
inline void Event<Args...>::Invoke(Args... args)
{
	for (auto const &[key, val] : m_subscribers)
	{
		val(args...);
	}
}
}