#pragma once

#include "TypeInfo.hpp"

namespace Ilum
{
class Enum : public TypeInfo
{
  public:
	struct Element
	{
		std::string name;
		std::string qualified_name;
	};

  public:
	Enum(const Cursor &cursor, const Namespace &current_namespace);

	virtual ~Enum() override = default;

	const std::string &GetName() const;

	const std::string &GetQualifiedName() const;

	const std::vector<Element> &GetElements() const;

	virtual kainjow::mustache::data GenerateReflection() const override;

  private:
	std::string          m_name;
	std::string          m_qualified_name;
	std::vector<Element> m_elements;
};
}        // namespace Ilum