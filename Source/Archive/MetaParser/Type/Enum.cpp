#include "Enum.hpp"

namespace Ilum
{
Enum::Enum(const Cursor &cursor, const Namespace &current_namespace) :
    TypeInfo(cursor, current_namespace), m_name(cursor.GetDisplayName()), m_qualified_name(cursor.GetType().GetDispalyName())
{
	for (auto &child : cursor.GetChildren())
	{
		if (child.GetKind() == CXCursor_EnumConstantDecl)
		{
			m_elements.push_back({child.GetDisplayName(), child.GetType().GetDispalyName() + "::" + child.GetDisplayName()});
		}
	}
}

const std::string &Enum::GetName() const
{
	return m_name;
}

const std::string &Enum::GetQualifiedName() const
{
	return m_qualified_name;
}

const std::vector<Enum::Element> &Enum::GetElements() const
{
	return m_elements;
}

kainjow::mustache::data Enum::GenerateReflection() const
{
	kainjow::mustache::data enum_data{kainjow::mustache::data::type::object};
	enum_data["EnumQualifiedName"] = m_qualified_name;
	enum_data["EnumName"]          = m_name;

	kainjow::mustache::data members{kainjow::mustache::data::type::list};

	for (size_t i = 0; i < m_elements.size(); i++)
	{
		kainjow::mustache::data member{kainjow::mustache::data::type::object};
		member["EnumValueName"]          = m_elements[i].name;
		member["EnumValueQualifiedName"] = m_elements[i].qualified_name;
		member["IsLast"]                 = i == (m_elements.size() - 1);
		members << member;
	}
	enum_data.set("EnumValue", members);
	return enum_data;
}
}        // namespace Ilum