#pragma once

#include <string>
#include <typeindex>

namespace Ilum
{
class Node;

class Component
{
  public:
	Component(const std::string &name, Node* node);

	Component(Component &&) = default;

	virtual ~Component() = default;

	virtual void OnImGui() = 0;

	virtual std::type_index GetType() const = 0;

	const std::string &GetName();

  protected:
	const std::string m_name;

	Node *p_node = nullptr;
};
}        // namespace Ilum