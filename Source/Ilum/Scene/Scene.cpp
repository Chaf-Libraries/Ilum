#include "Scene.hpp"

#include "EntityManager.hpp"
#include "SystemManager.hpp"

namespace Ilum
{
Scene::Scene(Context *context) :
    TSubsystem<Scene>(context),
    m_system_manager(createScope<SystemManager>()),
    m_entity_manager(createScope<EntityManager>())
{

}

void Scene::onPreTick()
{
}

void Scene::onTick(float delta_time)
{
	m_system_manager->run();
}

void Scene::clear()
{
	m_entity_manager->clear();
	name = "";
}

entt::registry &Scene::getRegistry()
{
	return m_entity_manager->getRegistry();
}

Entity Scene::createEntity()
{
	return m_entity_manager->create();
}

Entity Scene::createEntity(const std::string &name)
{
	return m_entity_manager->create(name);
}
}        // namespace Ilum