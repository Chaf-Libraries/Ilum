#pragma once

#include "Utils/PCH.hpp"

#include "Engine/Subsystem.hpp"

#include "SystemManager.hpp"

#include <entt.hpp>

namespace Ilum
{
class Entity;
class EntityManager;

class Scene : public TSubsystem<Scene>
{
  public:
	Scene(Context *context = nullptr);

	~Scene() = default;

	virtual void onPreTick() override;

	virtual void onTick(float delta_time) override;

	void clear();

	entt::registry &getRegistry();

	Entity createEntity();

	Entity createEntity(const std::string &name);

	template<typename T>
	void addSystem()
	{
		m_system_manager->add<T>();
	}

  public:
	std::string name = "";

  private:
	scope<EntityManager> m_entity_manager;
	scope<SystemManager> m_system_manager;
};
}        // namespace Ilum