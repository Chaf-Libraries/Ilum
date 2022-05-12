#pragma once

#include "Scene.hpp"

namespace Ilum
{
class Entity
{
  private:
	Scene &m_scene;

	entt::entity m_handle = entt::null;

  public:
	explicit Entity(Scene &scene, entt::entity handle = entt::null) :
	    m_scene(scene), m_handle(handle)
	{
	}

	~Entity() = default;

	Entity(const Entity &other) :
	    m_scene(other.m_scene),
	    m_handle(other.m_handle)
	{
	}

	Entity &operator=(const Entity &other)
	{
		ASSERT(&m_scene == &other.m_scene);
		m_handle = other.m_handle;
		return *this;
	}

	template <typename T, typename... Args>
	inline T &AddComponent(Args &&...args)
	{
		return m_scene.GetRegistry().emplace<T>(m_handle, std::forward<Args>(args)...);
	}

	template <typename T, typename... Args>
	inline T &GetOrAddComponent(Args &&...args)
	{
		return m_scene.GetRegistry().get_or_emplace<T>(m_handle, std::forward<Args>(args)...);
	}

	template <typename T, typename... Args>
	inline T &GetOrReplaceComponent(Args &&...args)
	{
		return m_scene.GetRegistry().emplace_or_replace<T>(m_handle, std::forward<Args>(args)...);
	}

	template <typename T>
	inline T &GetComponent()
	{
		return m_scene.GetRegistry().get<T>(m_handle);
	}

	template <typename T>
	inline T *TryGetComponent()
	{
		return m_scene.GetRegistry().try_get<T>(m_handle);
	}

	template <typename T, typename... Args>
	inline bool HasComponent()
	{
		return m_scene.GetRegistry().all_of<T, Args...>(m_handle);
	}

	template <typename T>
	inline void RemoveComponent()
	{
		m_scene.GetRegistry().remove<T>(m_handle);
	}

	template <typename T>
	inline void TryRemoveComponent()
	{
		if (HasComponent<T>())
		{
			RemoveComponent<T>();
		}
	}

	inline operator const entt::entity &() const
	{
		return m_handle;
	}

	inline operator uint32_t() const
	{
		return static_cast<uint32_t>(m_handle);
	}

	inline operator bool() const
	{
		return m_handle != entt::null && m_scene.GetRegistry().valid(m_handle);
	}

	inline bool operator==(const Entity &rhs) const
	{
		return m_handle == rhs.m_handle;
	}

	inline bool operator==(entt::entity rhs) const
	{
		return m_handle == rhs;
	}

	inline entt::entity GetHandle() const
	{
		return m_handle;
	}

	inline void Destroy()
	{
		if (IsValid())
		{
			m_scene.GetRegistry().destroy(m_handle);
		}
	}

	inline bool IsValid()
	{
		return m_scene.GetRegistry().valid(m_handle);
	}
};
}        // namespace Ilum

namespace std
{
inline std::string to_string(Ilum::Entity entity)
{
	return to_string(static_cast<uint32_t>(entity.GetHandle()));
}
}        // namespace std