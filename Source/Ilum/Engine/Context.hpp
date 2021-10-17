#pragma once

#include "Utils/PCH.hpp"
#include "Subsystem.hpp"

#include <vector>

namespace Ilum
{
class Context
{
  public:
	Context() = default;

	~Context();

	void onInitialize();
	void onPreTick();
	void onTick(TickType tick_group, float delta_time);
	void onPostTick();
	void onShutdown();

	template <typename T>
	void addSubsystem(TickType tick_group = TickType::Variable)
	{
		validateSubsystemType<T>();

		m_subsystems.emplace_back(std::make_unique<T>(this), tick_group);
	}

	template <typename T>
	bool hasSubsystem()
	{
		validateSubsystemType<T>();

		for (const auto &subsystem : m_subsystems)
		{
			if (subsystem.ptr)
			{
				if (typeid(T) == typeid(*subsystem.ptr))
				{
					return true;
				}
			}
		}
		return false;
	}

	template <typename T>
	T *getSubsystem()
	{
		validateSubsystemType<T>();

		for (const auto &subsystem : m_subsystems)
		{
			if (subsystem.ptr)
			{
				if (std::type_index(typeid(T)) == subsystem.ptr->type())
				{
					return static_cast<T *>(subsystem.ptr.get());
				}
			}
		}
		return nullptr;
	}

	template <typename T>
	bool removeSubsystem()
	{
		validateSubsystemType<T>();

		for (auto subsystem = m_subsystems.begin(); subsystem != m_subsystems.end(); subsystem++)
		{
			if (subsystem->ptr)
			{
				if (typeid(T) == typeid(*(subsystem->ptr)))
				{
					subsystem->ptr->onShutdown();
					m_subsystems.erase(subsystem);
					return true;
				}
			}
		}

		return false;
	}

	template <typename T>
	bool validateSubsystemType()
	{
		static_assert(std::is_base_of<ISubsystem, T>::value, "Provided type is not derived from ISubsystem!");
		return true;
	}

  private:
	std::vector<Subsystem> m_subsystems;
};
}        // namespace Tools