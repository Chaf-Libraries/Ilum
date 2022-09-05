#include "Meta.hpp"

#include <spdlog/fmt/fmt.h>

namespace Ilum
{
namespace Meta
{
std::string Attribute::GenerateName(bool is_string) const
{
	return is_string ? fmt::format("\"{}::{}\"", _namespace, name) :
                       _namespace + "::" + name;
}

std::string Parameter::GenerateTypeName() const
{
	std::string rst = type;
	if (is_packed)
	{
		rst += "...";
	}
	return rst;
}

std::string Parameter::GenerateParameterName() const
{
	std::string rst = GenerateTypeName();
	if (is_packed)
	{
		rst += "...";
	}
	rst += " ";
	rst += name;
	return rst;
}

std::string Parameter::GenerateArgumentName() const
{
	std::string rst = name;
	if (is_packed)
	{
		rst += "...";
	}
	return rst;
}

bool Field::IsStaticConstexprVariable() const
{
	bool contains_static    = false;
	bool contains_constexpr = false;
	for (const auto &decl_specifier : decl_specifiers)
	{
		if (decl_specifier == "static")
		{
			contains_static = true;
		}
		else if (decl_specifier == "constexpr")
		{
			contains_constexpr = true;
		}
	}
	return mode == Mode::Variable && contains_static && contains_constexpr;
}

bool Field::IsMemberFunction() const
{
	if (mode != Mode::Function)
	{
		return false;
	}

	bool contains_static = false;
	for (const auto &decl_specifier : decl_specifiers)
	{
		if (decl_specifier == "static")
		{
			contains_static = true;
		}
	}

	bool contains_friend = false;
	for (const auto &decl_specifier : decl_specifiers)
	{
		if (decl_specifier == "friend")
		{
			contains_friend = true;
		}
	}

	return !contains_static && !contains_friend;
}

bool Field::IsFriendFunction() const
{
	if (mode != Mode::Function)
	{
		return false;
	}

	for (const auto &decl_specifier : decl_specifiers)
	{
		if (decl_specifier == "friend")
		{
			return true;
		}
	}

	return false;
}

bool Field::IsDeleteFunction() const
{
	return mode == Mode::Function && initializer == "delete";
}

std::string Field::GenerateParameterTypeList() const
{
	return GenerateParameterTypeList(parameters.size());
}

std::string Field::GenerateParameterTypeList(size_t num) const
{
	std::string rst;
	for (std::size_t i = 0; i < num; i++)
	{
		rst += parameters[i].GenerateTypeName();
		if (i != num - 1)
		{
			rst += ", ";
		}
	}
	return rst;
}

std::string Field::GenerateReturnType() const
{
	if (mode == Mode::Function)
	{
		std::string rst = "";
		for (auto& decl_specifier : decl_specifiers)
		{
			if (decl_specifier != "virtual" &&	
				decl_specifier != "static")
			{
				rst += decl_specifier + " ";
			}
		}
		for (auto& pointer_operator : pointer_operators)
		{
			rst += pointer_operator + " ";
		}
		return rst;
	}

	return "";
}

bool Field::IsConstructor() const
{
	for (auto &attribute : attributes)
	{
		if (attribute.name == "constructor" && attribute.value == "true")
		{
			return true;
		}
	}

	return false;
}

bool Field::IsPureVirtual() const
{
	return initializer == "0";
}

bool Field::NoReflection() const
{
	for (auto &attribute : attributes)
	{
		if (attribute.name == "reflection" && attribute.value == "false")
		{
			return true;
		}
	}

	return false;
}

bool Field::NoSerialization() const
{
	for (auto &attribute : attributes)
	{
		if (attribute.name == "serialization" && attribute.value == "false")
		{
			return true;
		}
	}

	return false;
}

bool TypeMeta::IsTemplateType() const
{
	return !template_parameters.empty();
}

std::string TypeMeta::GenerateName() const
{
	std::string rst = "";
	for (const auto &ns : namespaces)
	{
		rst += ns;
		rst += "::";
	}
	rst += name;
	return rst;
}

std::string TypeMeta::GenerateFullName() const
{
	std::string rst = GenerateName();
	if (!IsTemplateType())
	{
		return rst;
	}

	size_t idx = 0;
	rst += "<";
	for (size_t i = 0; i < template_parameters.size(); i++)
	{
		const auto &ele = template_parameters[i];
		if (ele.name.empty())
		{
			rst += "_";
			rst += std::to_string(idx);
			idx++;
		}
		else
		{
			rst += ele.name;
		}
		if (ele.is_packed)
		{
			rst += "...";
		}
		if (i != template_parameters.size() - 1)
		{
			rst += ", ";
		}
	}
	rst += ">";

	return rst;
}

std::string TypeMeta::GenerateTemplateList() const
{
	std::string rst;

	size_t idx = 0;
	for (size_t i = 0; i < template_parameters.size(); i++)
	{
		const auto &ele = template_parameters[i];
		rst += ele.type;
		if (ele.is_packed)
		{
			rst += "...";
		}
		rst += " ";
		if (ele.name.empty())
		{
			rst += "_";
			rst += std::to_string(idx);
			idx++;
		}
		else
		{
			rst += ele.name;
		}
		if (i != template_parameters.size() - 1)
		{
			rst += ", ";
		}
	}

	return rst;
}

bool TypeMeta::NeedReflection() const
{
	for (auto &attribute : attributes)
	{
		if (attribute.name == "reflection" && attribute.value == "true")
		{
			return true;
		}
	}

	return false;
}

bool TypeMeta::NeedSerialization() const
{
	for (auto &attribute : attributes)
	{
		if (attribute.name == "serialization" && attribute.value == "true")
		{
			return true;
		}
	}

	return false;
}

bool TypeMeta::IsPureVirtual() const
{
	for (auto& field : fields)
	{
		if (field.IsPureVirtual())
		{
			return true;
		}
	}
	return false;
}

bool TypeMeta::HasConstructor() const
{
	for (auto &field : fields)
	{
		if (field.IsConstructor())
		{
			return true;
		}
	}
	return false;
}

bool TypeMeta::IsOverload(const std::string &name) const
{
	uint32_t count = 0;
	for (auto &field : fields)
	{
		if (field.name == name)
		{
			count++;
		}
	}
	return count > 1;
}
}        // namespace Meta
}        // namespace Ilum