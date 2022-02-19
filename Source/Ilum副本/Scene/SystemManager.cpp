#include "SystemManager.hpp"

namespace Ilum
{
void SystemManager::run()
{
	std::vector<System *> used_systems;
	for (auto &sym : m_systems)
	{
		sym->run();
	}
}

void SystemManager::clear()
{
	m_systems.clear();
}
}        // namespace Ilum