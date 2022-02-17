#pragma once

#include <functional>
#include <unordered_map>

namespace Ilum::Core
{
template <typename... Args>
class Event
{
  public:
	using CallbackFunc = std::function<void(Args...)>;

	uint32_t Subscribe(CallbackFunc callback);

	uint32_t operator+=(CallbackFunc callback);

	bool Unsubscribe(uint32_t subscriber);

	bool operator-=(uint32_t subscriber);

	void Clear();

	uint32_t GetSubscribersCount() const;

	void Invoke(Args... args);

  private:
	std::unordered_map<uint32_t, CallbackFunc> m_subscribers;
	uint32_t                                   m_avaliable_id = 0;
};
}        // namespace Ilum::Core

#include "Event.inl"