#pragma once

#include "Scene.hpp"

namespace Ilum
{
class Entity
{
  public:
	explicit Entity(Scene *scene, entt::entity handle);

	~Entity() = default;

	template <typename T, typename... Args>
	T &AddComponent(Args &&...args)
	{
		return p_scene->m_registry.emplace<T>(m_handle, std::forward<Args>(args)...);
	}

	template <typename T>
	T &GetComponent()
	{
		return p_scene->m_registry.get<T>(m_handle);
	}

	template <typename T, typename... Args>
	bool HasComponent()
	{
		return p_scene->m_registry.all_of<T, Args...>(m_handle);
	}

	template <typename T>
	void RemoveComponent()
	{
		if (HasComponent<T>())
		{
			p_scene->m_registry.remove<T>(m_handle);
		}
	}

	operator bool() const;

	bool operator==(const Entity &rhs) const;

	bool operator==(entt::entity rhs) const;

	entt::entity GetHandle() const;

	void Destroy();

	bool IsValid() const;

  private:
	entt::entity m_handle = entt::null;

	Scene *p_scene = nullptr;
};
}        // namespace Ilum