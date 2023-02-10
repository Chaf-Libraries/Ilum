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

	virtual ~Component();

	virtual bool OnImGui() = 0;

	virtual std::type_index GetType() const = 0;

	const char *GetName() const;

	Node *GetNode() const;

	virtual void Save(OutputArchive &archive) const = 0;

	virtual void Load(InputArchive &archive) = 0;

	bool IsUpdate();

	void SetUpdate(bool update);

  protected:
	const char *m_name;

	bool m_update = false;

	Node *p_node = nullptr;
};
}        // namespace Ilum