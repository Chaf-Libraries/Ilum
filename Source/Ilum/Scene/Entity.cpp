#include "Entity.hpp"
#include "Scene.hpp"

namespace Ilum
{
Entity::Entity(Scene *scene, entt::entity handle) :
    p_scene(scene), m_handle(handle)
{
}

Entity::operator bool() const
{
	return IsValid();
}

bool Entity::operator==(const Entity& rhs) const
{
	return m_handle == rhs.m_handle;
}

bool Entity::operator==(entt::entity rhs) const
{
	return m_handle == rhs;
}

entt::entity Entity::GetHandle() const
{
	return m_handle;
}

void Entity::Destroy()
{
	p_scene->m_registry.destroy(m_handle);
	m_handle = entt::null;
}

bool Entity::IsValid() const
{
	return p_scene && p_scene->m_registry.valid(m_handle);
}
}        // namespace Ilum