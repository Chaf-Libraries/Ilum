#include "Method.hpp"

namespace Ilum
{
Method::Method(const Cursor &cursor, const Namespace &current_namespace, Class *parent, bool is_static, bool is_constructor) :
    TypeInfo(cursor, current_namespace), m_name(cursor.GetDisplayName()), m_qualified_name(cursor.GetType().GetDispalyName()), m_parent(parent), m_is_static(is_static), m_is_constructor(is_constructor)
{
	m_name = m_name.substr(0, m_name.find_first_of("("));

	m_return_type = m_qualified_name.substr(0, m_qualified_name.find_first_of("("));

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

bool Method::ShouldCompile() const
{
	return true;
}

bool Method::ShouldReflection() const
{
	return true;
}

bool Method::IsConstructor() const
{
	return IsMemberMethod() && m_is_constructor;
}

bool Method::IsStatic() const
{
	return m_is_static;
}

bool Method::IsMemberMethod() const
{
	return m_parent != nullptr;
}

kainjow::mustache::data Method::GenerateReflection() const
{
	kainjow::mustache::data method_data{kainjow::mustache::data::type::object};
	method_data["Constructor"] = false;

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
		method_data.set("Params", constructor_params);
		method_data["Constructor"] = true;
		return method_data;
	}

	return method_data;
}
}        // namespace Ilum