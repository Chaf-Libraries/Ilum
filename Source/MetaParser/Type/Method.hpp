#pragma once

#include "TypeInfo.hpp"

namespace Ilum
{
class Class;

class Method : public TypeInfo
{
  public:
	using Parameter = std::pair<std::string, std::string>;

  public:
	Method(const Cursor &cursor, const Namespace &current_namespace, Class *parent = nullptr, bool is_static = false, bool is_constructor = false);

	virtual ~Method() = default;

	virtual bool ShouldReflection() const override;

	bool IsConstructor() const;

	bool IsStatic() const;

	bool IsMemberMethod() const;

	bool IsPureVirtualMethod() const;

	const std::string &GetName() const;

	virtual kainjow::mustache::data GenerateReflection() const override;

  private:
	std::string m_name;

	std::string m_qualified_name;

	std::vector<Parameter> m_parameters;

	std::string m_return_type = "";

	Class *m_parent = nullptr;

	bool m_is_const;

	bool m_is_static;

	bool m_is_constructor;

	bool m_is_pure_virtual;
};
}        // namespace Ilum