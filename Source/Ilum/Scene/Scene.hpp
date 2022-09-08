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

	void Load(const std::string &filename);

	void Save(const std::string &filename);

	entt::registry &operator()();

  private:
	std::string m_name;

	entt::registry m_registry;
};
}        // namespace Ilum