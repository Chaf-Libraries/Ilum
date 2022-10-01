#pragma once

#include "TypeInfo.hpp"

namespace Ilum
{
class Field;
class Method;
class MetaParser;

struct BaseClass
{
	BaseClass(const Cursor &cursor);

	std::string name;
};

class Class : public TypeInfo
{
  public:
	Class(MetaParser *parser, const Cursor &cursor, const Namespace &current_namespace);

	std::string GetClassName();

	const std::string GetName() const;

	const std::string GetQualifiedName() const;

	const std::vector<std::shared_ptr<Field>> &GetFields() const;

	const std::vector<std::shared_ptr<Method>> &GetMethods() const;

	bool IsMethodOverloaded(const std::string& method) const;

	bool HasConstructor() const;

	virtual kainjow::mustache::data GenerateReflection() const override;

  private:
	std::vector<std::shared_ptr<BaseClass>> m_base_classes;

	std::string m_name;

	std::string m_qualified_name;

	std::vector<std::shared_ptr<Field>> m_fields;
	std::vector<std::shared_ptr<Method>> m_methods;

	bool m_has_constructor = false;

	std::unordered_map<std::string, uint32_t> m_overload;

	bool IsAccessible() const;
};
}        // namespace Ilum