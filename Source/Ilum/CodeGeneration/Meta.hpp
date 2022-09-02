#pragma once

#include <string>
#include <vector>

namespace Ilum::Meta
{
struct Attribute
{
	std::string _namespace;
	std::string name;
	std::string value;
};

enum class AccessSpecifier
{
	Public,
	Private,
	Protected,
	Default
};

using DeclSpecifier = std::string;

struct Parameter
{
	std::string name;
	std::string type;
};

struct Field
{
	enum class Mode
	{
		Variable,
		Function,
		Value
	};

	std::string            name;
	Mode        mode;
	AccessSpecifier        specifier;
	std::vector<Attribute> attributes;
};

struct MetaType
{
	enum class Mode
	{
		Enum,
		Class,
		Struct
	};

	Mode                     mode;
	std::vector<std::string> _namespaces;
	std::string              name;
	std::vector<Attribute>   attributes;
	std::vector<Field>       fields;
	//bool                     is_template = false;
};
}        // namespace Ilum::Meta