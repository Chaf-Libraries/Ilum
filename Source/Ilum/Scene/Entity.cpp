#include "Entity.hpp"

#include "Component/Tag.hpp"

#include "Scene.hpp"

namespace Ilum
{
Entity::Entity(entt::entity handle):
    m_handle(handle)
{

}

bool Entity::active() const
{
	return m_active;
}

void Entity::setActive(bool is_active)
{
	m_active = is_active;
}

Entity::operator const entt::entity &() const
{
	return m_handle;
}

Entity::operator uint32_t() const
{
	return static_cast<uint32_t>(m_handle);
}

Entity::operator bool() const
{
	return m_handle != entt::null;
}

entt::entity Entity::getHandle() const
{
	return m_handle;
}

void Entity::destroy()
{
	Scene::instance()->getRegistry().destroy(m_handle);
}

bool Entity::valid()
{
	return Scene::instance()->getRegistry().valid(m_handle);
}

}        // namespace Ilum