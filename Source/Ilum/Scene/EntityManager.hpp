#pragma once

#include <entt.hpp>

#include "Entity.hpp"

namespace Ilum
{
class EntityManager
{
  public:
	EntityManager() = default;

	~EntityManager();

	Entity create();

	Entity create(const std::string &name);

	void clear();

	Entity getEntityByUUID(uint64_t uuid);

  private:
	entt::registry m_registry;
};
}        // namespace Ilum