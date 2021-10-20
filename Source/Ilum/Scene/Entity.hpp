#pragma once

#include <entt.hpp>

#include "Eventing/Event.hpp"

namespace Ilum
{
class Entity
{
  public:
	Entity() = default;

	Entity(entt::entity handle);

	~Entity() = default;

	template <typename T, typename... Args>
	T &addComponent(Args &&...args);

	template <typename T, typename... Args>
	T &getOrAddComponent(Args &&...args);

	template <typename T, typename... Args>
	T &getOrReplaceComponent(Args &&...args);

	template <typename T>
	T &getComponent();

	template <typename T>
	T *tryGetComponent();

	template <typename T>
	bool hasComponent();

	template <typename T>
	void removeComponent();

	template <typename T>
	void tryRemoveComponent();

	bool active() const;

	void setActive(bool is_active);

	const std::string &getName() const;

	void setName(const std::string &name);

	bool hasParent() const;

	bool hasChildren() const;

	Entity getParent() const;

	std::vector<Entity> getChildren() const;

	void setParent(Entity entity) const;

	void clearChildren(Entity entity);

	void addChildren(Entity entity);

	operator const entt::entity &() const;

	operator uint32_t() const;

	operator bool() const;

	entt::entity getHandle() const;

	void destroy();

	bool valid();

  private:
	entt::entity m_handle = entt::null;

  public:
	static Event<> Event_Add;
};
}        // namespace Ilum