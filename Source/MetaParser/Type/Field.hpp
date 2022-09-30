#pragma once

#include "Class.hpp"
#include "TypeInfo.hpp"

namespace Ilum
{
class Field : public TypeInfo
{
  public:
	Field(const Cursor &cursor, const Namespace &current_namespace, Class *parent = nullptr);

	virtual ~Field() = default;

	const std::string &GetName() const;

	std::string GetQualifiedName() const;

	virtual kainjow::mustache::data GenerateReflection() const override;

  private:
	bool m_is_const;

	Class *m_parent;

	std::string m_name;

	std::string m_type;

	std::string m_default;

	bool IsAccessible() const;
};
}        // namespace Ilum