#include "Scene.hpp"
#include "Component/HierarchyComponent.hpp"
#include "Component/AllComponent.hpp"
#include "Entity.hpp"

namespace Ilum
{
Scene::Scene(const std::string &name) :
    m_name(name)
{
}

Scene::~Scene()
{
	Clear();
	m_registry.clear();
}

void Scene::Tick()
{
	// System<StaticMeshComponent>().Tick(this);
}

Entity Scene::CreateEntity(const std::string &name)
{
	auto entity = m_registry.create();

	m_registry.emplace<TagComponent>(entity).tag = name;
	m_registry.emplace<TransformComponent>(entity);
	m_registry.emplace<HierarchyComponent>(entity);

	return Entity(this, (uint32_t) entity);
}

void Scene::Clear()
{
	m_registry.each([&](auto entity) { m_registry.destroy(entity); });
}

void Scene::Execute(std::function<void(Entity &)> &&func)
{
	m_registry.each([&](entt::entity entity_id) {
		auto e = Entity(this, (uint32_t) entity_id);
		func(e);
	});
}

void Scene::SetName(const std::string &name)
{
	m_name = name;
}

const std::string &Scene::GetName() const
{
	return m_name;
}

entt::registry &Scene::operator()()
{
	return m_registry;
}

size_t Scene::Size() const
{
	return m_registry.size();
}
}        // namespace Ilum