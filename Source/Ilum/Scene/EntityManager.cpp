#include "EntityManager.hpp"

#include "Component/Tag.hpp"
#include "Component/UUID.hpp"

namespace Ilum
{
EntityManager::~EntityManager()
{
}

Entity EntityManager::create()
{
	auto entity = m_registry.create();
	m_registry.emplace<Cmpt::UUID>(entity);
	return Entity(entity);
}

Entity EntityManager::create(const std::string &name)
{
	auto entity = m_registry.create();
	m_registry.emplace<Cmpt::UUID>(entity);
	m_registry.emplace<Cmpt::Tag>(entity, name);
	return Entity(entity);
}

void EntityManager::clear()
{
	m_registry.each([&](auto entity) { m_registry.destroy(entity); });
}

Entity EntityManager::getEntityByUUID(uint64_t uuid)
{
	auto view = m_registry.view<Cmpt::UUID>();
	for (auto& entity : view)
	{
		auto &uuid_cmpt = m_registry.get<Cmpt::UUID>(entity);
		if (uuid_cmpt.id == uuid)
		{
			return Entity(entity);
		}
	}

	return Entity();
}
}        // namespace Ilum