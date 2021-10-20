#pragma once

#include "Entity.hpp"
#include "Scene/Scene.hpp"

namespace Ilum
{
template <typename T, typename... Args>
inline T &Entity::addComponent(Args &&...args)
{
	ASSERT(!hasComponent<T>());
	return Scene::instance()->getRegistry().emplace<T>(m_handle, std::forward<Args>(args)...);
}

template <typename T, typename... Args>
inline T &Entity::getOrAddComponent(Args &&...args)
{
	return Scene::instance()->getRegistry().get_or_emplace<T>(m_handle, std::forward<Args>(args)...);
}

template <typename T, typename... Args>
inline T &Entity::getOrReplaceComponent(Args &&...args)
{
	return Scene::instance()->getRegistry().emplace_or_replace<T>(m_handle, std::forward<Args>(args)...);
}

template <typename T>
inline T &Entity::getComponent()
{
	return Scene::instance()->getRegistry().get<T>(m_handle);
}

template <typename T>
inline T *Entity::tryGetComponent()
{
	return Scene::instance()->getRegistry().try_get<T>(m_handle);
}

template <typename T, typename... Args>
inline bool Entity::hasComponent()
{
	return Scene::instance()->getRegistry().all_of<T, Args...>(m_handle);
}

template <typename T>
inline void Entity::removeComponent()
{
	Scene::instance()->getRegistry().remove<T>(m_handle);
}

template <typename T>
inline void Entity::tryRemoveComponent()
{
	if (hasComponent<T>())
	{
		removeComponent<T>();
	}
}
}