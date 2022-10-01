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
	void GroupExecute(std::function<void(uint32_t, Tn &...)> &&func)
	{
		m_registry.view<Tn...>().each([&](entt::entity entity, Tn &...tn) {
			func((uint32_t) entity, tn...);
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