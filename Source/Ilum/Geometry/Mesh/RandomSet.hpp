#pragma once

#include "Utils/PCH.hpp"

namespace Ilum
{
template <typename _Ty>
class RandomSet
{
  public:
	RandomSet()  = default;
	~RandomSet() = default;

	void insert(_Ty data)
	{
		assert(!has(data));
		m_lookup.emplace(data, m_data.size());
		m_data.push_back(data);
	}

	void erase(_Ty data)
	{
		auto target = m_lookup.find(data);
		assert(target != m_lookup.end());
		auto idx = target->second;
		if (idx != m_data.size() - 1)
		{
			m_lookup[m_data.back()] = idx;
			m_data[idx]             = m_data.back();
		}
		m_data.pop_back();
		m_lookup.erase(target);
	}

	auto begin() noexcept
	{
		return m_data.begin();
	}

	auto begin() const noexcept
	{
		return m_data.begin();
	}

	auto end() noexcept
	{
		return m_data.end();
	}

	auto end() const noexcept
	{
		return m_data.end();
	}

	_Ty operator[](size_t idx)
	{
		return m_data[idx];
	}

	_Ty at(size_t idx)
	{
		return m_data.at(idx);
	}

	size_t size() const noexcept
	{
		return m_data.size();
	}

	void reserve(size_t n)
	{
		m_data.reserve(n);
		m_lookup.reserve(n);
	}

	void clear() noexcept
	{
		m_data.clear();
		m_lookup.clear();
	}

	const std::vector<_Ty> &vec() const noexcept
	{
		return m_data;
	}

	size_t idx(_Ty data) const
	{
		return m_lookup.at(data);
	}

	bool has(_Ty data) const
	{
		return m_lookup.find(data) != m_lookup.end();
	}

	bool empty() const noexcept
	{
		return m_data.empty();
	}

  private:
	std::unordered_map<_Ty, std::size_t> m_lookup;
	std::vector<_Ty>                     m_data;
};
}        // namespace Ilum