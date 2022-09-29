#include "Class.hpp"
#include "Field.hpp"
#include "Method.hpp"

namespace Ilum
{
BaseClass::BaseClass(const Cursor &cursor) :
    name(cursor.GetType().GetDispalyName())
{
}

Class::Class(const Cursor &cursor, const Namespace &current_namespace) :
    TypeInfo(cursor, current_namespace), m_name(cursor.GetDisplayName()), m_qualified_name(cursor.GetType().GetDispalyName())
{
	for (auto &child : cursor.GetChildren())
	{
		switch (child.GetKind())
		{
			case CXCursor_CXXBaseSpecifier:
				m_base_classes.emplace_back(std::make_shared<BaseClass>(child));
				break;
			case CXCursor_FieldDecl:
				m_fields.emplace_back(std::make_shared<Field>(child, current_namespace, this));
				break;
			case CXCursor_Constructor:
				m_methods.emplace_back(std::make_shared<Method>(child, current_namespace, this, false, true));
				break;
			case CXCursor_CXXMethod:
				m_methods.emplace_back(std::make_shared<Method>(child, current_namespace, this, false));
				break;
			default:
				break;
		}
	}
}

bool Class::ShouldCompile() const
{
	return true;
}

bool Class::ShouldReflection() const
{
	return GetMetaData().GetFlag(NativeProperty::Reflection);
}

std::string Class::GetClassName()
{
	return m_name;
}

const std::string Class::GetName() const
{
	return m_name;
}

const std::string Class::GetQualifiedName() const
{
	return m_qualified_name;
}

const std::vector<std::shared_ptr<Field>> &Class::GetFields() const
{
	return m_fields;
}

const std::vector<std::shared_ptr<Method>> &Class::GetMethods() const
{
	return m_methods;
}

kainjow::mustache::data Class::GenerateReflection() const
{
	kainjow::mustache::data class_data{kainjow::mustache::data::type::object};
	class_data["ClassQualifiedName"] = m_qualified_name;
	class_data["ClassName"]          = m_name;

	// Field
	{
		kainjow::mustache::data field_list{kainjow::mustache::data::type::list};
		for (auto &field : m_fields)
		{
			field_list << field->GenerateReflection();
		}
		class_data.set("Field", field_list);
	}

	// Method
	{
		kainjow::mustache::data method_list{kainjow::mustache::data::type::list};
		for (auto &method : m_methods)
		{
			method_list << method->GenerateReflection();
		}
		class_data.set("Method", method_list);
	}

	return class_data;
}

bool Class::IsAccessible() const
{
	return true;
}

}        // namespace Ilum