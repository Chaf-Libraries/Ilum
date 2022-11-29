#pragma once

#include "Precompile.hpp"

namespace Ilum
{
using DelegateHandle = uint32_t;

template <typename... Args>
class EXPORT_API MulticastDelegate
{
  public:
	using Delegate = std::function<void(Args...)>;

	DelegateHandle Subscribe(Delegate &&delegate)
	{
		while (IsLock())
		{
			std::this_thread::sleep_for(std::chrono::milliseconds(16));
		}

		DelegateHandle id = m_avaliable_id++;
		m_subscribers.emplace(id, std::move(delegate));
		return id;
	}

	DelegateHandle operator+=(Delegate &&delegate)
	{
		return Subscribe(std::move(delegate));
	}

	bool UnSubscribe(const DelegateHandle &handle)
	{
		while (IsLock())
		{
			std::this_thread::sleep_for(std::chrono::milliseconds(16));
		}

		if (m_subscribers.find(handle) != m_subscribers.end())
		{
			m_subscribers.erase(handle);
			return true;
		}
		return false;
	}

	bool operator-=(const DelegateHandle &handle)
	{
		return UnSubscribe(handle);
	}

	void Clear()
	{
		while (IsLock())
		{
			std::this_thread::sleep_for(std::chrono::milliseconds(16));
		}

		m_subscribers.clear();
	}

	void Invoke(Args... args)
	{
		Lock();
		for (auto const &[key, val] : m_subscribers)
		{
			val(args...);
		}
		Unlock();
	}

  private:
	void Lock()
	{
		m_lock++;
	}

	void Unlock()
	{
		assert(m_lock > 0);
		m_lock--;
	}

	bool IsLock()
	{
		return m_lock > 0;
	}

  private:
	std::unordered_map<DelegateHandle, Delegate> m_subscribers;

	DelegateHandle m_avaliable_id = 0;

	int32_t m_lock = 0;
};

}        // namespace Ilum