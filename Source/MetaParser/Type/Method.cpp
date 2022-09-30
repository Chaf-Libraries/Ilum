#include "Method.hpp"
#include "Class.hpp"
#include "Meta/MetaConfig.hpp"

namespace Ilum
{
Method::Method(const Cursor &cursor, const Namespace &current_namespace, Class *parent, bool is_static, bool is_constructor) :
    TypeInfo(cursor, current_namespace), m_name(cursor.GetDisplayName()), m_qualified_name(cursor.GetType().GetDispalyName()), m_parent(parent), m_is_static(is_static), m_is_constructor(is_constructor)
{
	m_name = m_name.substr(0, m_name.find_first_of("("));

	m_return_type = m_qualified_name.substr(0, m_qualified_name.find_first_of("("));

	m_is_const = m_qualified_name.substr(m_qualified_name.find_first_of(")")).find("const") < m_qualified_name.length();

	while (m_return_type.back() == ' ')
	{
		m_return_type.pop_back();
	}

	for (auto &child : cursor.GetChildren())
	{
		if (child.GetKind() == CXCursor_ParmDecl)
		{
			m_parameters.emplace_back(std::make_pair(child.GetType().GetDispalyName(), child.GetDisplayName()));
		}
	}
}

bool Method::IsConstructor() const
{
	return IsMemberMethod() && (m_is_constructor || m_meta_info.GetFlag(NativeProperty::Constructor));
}

bool Method::IsStatic() const
{
	return m_is_static;
}

bool Method::IsMemberMethod() const
{
	return m_parent != nullptr;
}

const std::string &Method::GetName() const
{
	return m_name;
}

kainjow::mustache::data Method::GenerateReflection() const
{
	kainjow::mustache::data method_data{kainjow::mustache::data::type::object};
	if (m_meta_info.Empty())
	{
		method_data["Meta"] = false;
	}
	else
	{
		method_data.set("Meta", m_meta_info.GenerateReflection());
	}

	if (m_is_constructor)
	{
		kainjow::mustache::data constructor_params{kainjow::mustache::data::type::list};
		for (size_t i = 0; i < m_parameters.size(); i++)
		{
			kainjow::mustache::data param{kainjow::mustache::data::type::object};
			param["Param"]  = m_parameters[i].first;
			param["IsLast"] = i == m_parameters.size() - 1;
			constructor_params << param;
		}
		method_data["Static"] = false;
		method_data.set("Params", constructor_params);
		return method_data;
	}

	if (m_meta_info.GetFlag(NativeProperty::Constructor))
	{
		kainjow::mustache::data constructor_params{kainjow::mustache::data::type::list};
		method_data["MethodQualifiedName"] = m_parent->GetQualifiedName() + "::" + m_name;
		method_data["Static"]              = true;
		method_data.set("Params", constructor_params);
		return method_data;
	}

	method_data["MethodQualifiedName"] = m_parent->GetQualifiedName() + "::" + m_name;
	method_data["MethodName"]          = m_name;
	method_data["ReturnType"]          = m_return_type;
	method_data["Const"]               = m_is_const;
	method_data["ClassQualifiedName"]  = m_parent->GetQualifiedName();
	kainjow::mustache::data method_params{kainjow::mustache::data::type::list};
	for (size_t i = 0; i < m_parameters.size(); i++)
	{
		kainjow::mustache::data param{kainjow::mustache::data::type::object};
		param["Param"]  = m_parameters[i].first;
		param["IsLast"] = i == m_parameters.size() - 1;
		method_params << param;
	}
	method_data.set("Params", method_params);

	return method_data;
}
}        // namespace Ilum