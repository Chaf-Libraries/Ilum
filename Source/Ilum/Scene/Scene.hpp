#pragma once

#include <entt.hpp>

namespace Ilum
{
class Entity;

class Scene
{
	friend class Entity;

  public:
	Scene(const std::string &name);

	~Scene();

	void Tick();

	Entity CreateEntity(const std::string &name = "New Entity");

	void Clear();

	void Execute(std::function<void(Entity &)> &&func);

	template <typename T>
	void GroupExecute(std::function<void(entt::entity entity, T &)> &&func)
	{
		m_registry.view<T>().each([&](auto entity, T &t) {
			func(entity, t);
		});
	}

	template <typename T1, typename... Tn>
	void GroupExecute(std::function<void(entt::entity entity, T1 &, Tn &...)> &&func)
	{
		m_registry.view<T1, Tn...>().each([&](auto entity, T1 &t1, Tn &...tn) {
			func(entity, t1, tn...);
		});
	}

	void SetName(const std::string &name);

	const std::string &GetName() const;

	entt::registry &operator()();

  private:
	std::string m_name;

	entt::registry m_registry;
};
}        // namespace Ilum