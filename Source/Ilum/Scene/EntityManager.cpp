#include "EntityManager.hpp"

#include "Component/Tag.hpp"

namespace Ilum
{
EntityManager::~EntityManager()
{
	clear();
}

Entity EntityManager::create()
{
	auto entity = m_registry.create();
	return Entity(entity);
}

Entity EntityManager::create(const std::string &name)
{
	auto entity = m_registry.create();
	m_registry.emplace<cmpt::Tag>(entity, name);
	return Entity(entity);
}

void EntityManager::clear()
{
	m_registry.each([&](auto entity) { m_registry.destroy(entity); });
}

entt::registry &EntityManager::getRegistry()
{
	return m_registry;
}
}        // namespace Ilum