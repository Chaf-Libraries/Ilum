#include "Meta.hpp"

#include <spdlog/fmt/fmt.h>

namespace Ilum::Meta
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
}        // namespace Ilum::Meta