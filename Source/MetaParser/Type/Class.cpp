#include "Class.hpp"
#include "Field.hpp"
#include "Method.hpp"
#include "Parser/Parser.hpp"

namespace Ilum
{
BaseClass::BaseClass(const Cursor &cursor) :
    name(cursor.GetType().GetDispalyName())
{
}

Class::Class(MetaParser *parser, const Cursor &cursor, const Namespace &current_namespace) :
    TypeInfo(cursor, current_namespace), m_name(cursor.GetDisplayName()), m_qualified_name(cursor.GetType().GetDispalyName())
{
	bool need_recursion = false;
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
				m_overload[m_methods.back()->GetName()]++;
				break;
			case CXCursor_EnumDecl:
			case CXCursor_StructDecl:
			case CXCursor_ClassDecl:
				need_recursion = true;
				break;
			default:
				break;
		}
	}

	if (need_recursion)
	{
		Namespace class_namespace = current_namespace;
		class_namespace.push_back(m_name);
		parser->BuildClassAST(cursor, class_namespace);
	}

	for (auto& method : m_methods)
	{
		if (method->IsConstructor())
		{
			m_has_constructor = true;
			return;
		}
	}
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

bool Class::IsMethodOverloaded(const std::string &method) const
{
	return m_overload.at(method) > 1;
}

bool Class::HasConstructor() const
{
	return m_has_constructor;
}

kainjow::mustache::data Class::GenerateReflection() const
{
	kainjow::mustache::data class_data{kainjow::mustache::data::type::object};
	class_data["ClassQualifiedName"] = m_qualified_name;
	class_data["ClassName"]          = m_name;
	class_data["NoConstructor"]          = !m_has_constructor;

	if (m_meta_info.Empty())
	{
		class_data["Meta"] = false;
	}
	else
	{
		class_data.set("Meta", m_meta_info.GenerateReflection());
	}

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
		kainjow::mustache::data constructor_list{kainjow::mustache::data::type::list};
		for (auto &method : m_methods)
		{
			if (method->IsConstructor())
			{
				constructor_list << method->GenerateReflection();
			}
			else
			{
				auto method_data        = method->GenerateReflection();
				method_data["Overload"] = IsMethodOverloaded(method->GetName());
				method_list << method_data;
			}
		}
		class_data.set("Method", method_list);
		class_data.set("Constructor", constructor_list);
	}

	return class_data;
}

bool Class::IsAccessible() const
{
	return true;
}

}        // namespace Ilum