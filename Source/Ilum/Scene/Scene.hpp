#pragma once

#include "Utils/PCH.hpp"

#include "Engine/Subsystem.hpp"

#include <entt.hpp>

namespace Ilum
{
class Entity;
class EntityManager;
class SystemManager;

class Scene : public TSubsystem<Scene>
{
  public:
	Scene(Context *context = nullptr);

	~Scene() = default;

	virtual void onTick(float delta_time) override;

	void clear();

	entt::registry &getRegistry();

	Entity createEntity();

	Entity createEntity(const std::string &name);

  private:
	std::string m_name = "untitled_scene";

	scope<EntityManager> m_entity_manager;
	scope<SystemManager> m_system_manager;
};
}        // namespace Ilum