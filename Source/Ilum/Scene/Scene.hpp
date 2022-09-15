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

	template <typename... Tn>
	void GroupExecute(std::function<void(entt::entity entity, Tn &...)> &&func)
	{
		m_registry.view<Tn...>().each([&](auto entity, Tn &...tn) {
			func(entity, tn...);
		});
	}

	void SetName(const std::string &name);

	const std::string &GetName() const;

	entt::registry &operator()();

	size_t Size() const;

  private:
	std::string m_name;

	entt::registry m_registry;
};
}        // namespace Ilum