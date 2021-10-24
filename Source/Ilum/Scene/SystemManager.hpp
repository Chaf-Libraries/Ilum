#pragma once

#include "Utils/PCH.hpp"

#include "System.hpp"

namespace Ilum
{
class SystemManager
{
  public:
	SystemManager() = default;

	~SystemManager() = default;

	template <typename T>
	void add()
	{
		static_assert(std::is_base_of_v<System, T>);
		m_systems.push_back(createScope<T>());
	}

	void run();

	void clear();

  private:
	std::vector<scope<System>> m_systems;
};
}        // namespace Ilum