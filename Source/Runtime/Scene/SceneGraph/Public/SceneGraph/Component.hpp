#pragma once

#include <Core/Core.hpp>

#include <string>
#include <typeindex>

namespace Ilum
{
class Node;

class EXPORT_API Component
{
  public:
	Component(const char *name, Node *node);

	Component(Component &&) = default;

	virtual ~Component() = default;

	virtual void OnImGui() = 0;

	virtual std::type_index GetType() const = 0;

	const char *GetName() const;

	Node *GetNode() const;

  private:
	const char *m_name;

	Node *p_node = nullptr;
};
}        // namespace Ilum