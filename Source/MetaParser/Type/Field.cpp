#include "Field.hpp"
#include "Class.hpp"

namespace Ilum
{
Field::Field(const Cursor &cursor, const Namespace &current_namespace, Class *parent) :
    TypeInfo(cursor, current_namespace), m_is_const(cursor.GetType().IsConstant()), m_parent(parent), m_name(cursor.GetSpelling()), m_type(cursor.GetType().GetDispalyName())
{
}

const std::string &Field::GetName() const
{
	return m_name;
}

std::string Field::GetQualifiedName() const
{
	return m_parent->GetClassName() + "::" + m_name;
}

kainjow::mustache::data Field::GenerateReflection() const
{
	kainjow::mustache::data field{kainjow::mustache::data::type::object};
	field["FieldName"]          = m_name;
	field["FieldQualifiedName"] = m_parent->GetQualifiedName() + "::" + m_name;

	if (m_meta_info.Empty())
	{
		field["Meta"] = false;
	}
	else
	{
		field.set("Meta", m_meta_info.GenerateReflection());
	}
	return field;
}

bool Field::IsAccessible() const
{
	return false;
}
}        // namespace Ilum