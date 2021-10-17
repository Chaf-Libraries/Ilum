#pragma once

#include "Utils/PCH.hpp"

namespace Ilum
{
template <typename... Args>
class Event
{
  public:
	using CallbackFunc = std::function<void(Args...)>;

	uint32_t subscribe(CallbackFunc callback);

	uint32_t operator+=(CallbackFunc callback);

	bool unsubscribe(uint32_t subscriber);

	bool operator-=(uint32_t subscriber);

	void clear();

	uint32_t getSubscribersCount() const;

	void invoke(Args... args);

  private:
	std::unordered_map<uint32_t, CallbackFunc> m_subscribers;
	uint32_t                               m_avaliable_id = 0;
};
}

#include "Event.inl"