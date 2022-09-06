#pragma once

#include <string>
#include <vector>

namespace Ilum
{
namespace Meta
{
struct Attribute
{
	std::string _namespace;
	std::string name;
	std::string value;

	std::string GenerateName(bool is_string) const;
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
	std::string type;        // typename, class, ...
	std::string name;
	std::string initializer;
	bool        is_packed = false;

	std::string GenerateTypeName() const;
	std::string GenerateParameterName() const;
	std::string GenerateArgumentName() const;
};

struct Field
{
	enum class Mode
	{
		Variable,
		Function,
		Value
	};

	Mode                       mode             = Mode::Variable;
	AccessSpecifier            access_specifier = AccessSpecifier::Private;
	std::vector<Attribute>     attributes;
	std::vector<DeclSpecifier> decl_specifiers;
	std::vector<std::string>   pointer_operators;        // *, &, &&
	std::string                name;
	std::string                initializer;        // expression or {expression}
	std::vector<Parameter>     parameters;
	std::vector<std::string>   qualifiers;        // const, volatile, &, &&
	bool                       is_template = false;

	bool        IsStaticConstexprVariable() const;
	bool        IsMemberFunction() const;
	bool        IsFriendFunction() const;
	bool        IsDeleteFunction() const;
	std::string GenerateParameterTypeList() const;
	std::string GenerateParameterTypeList(size_t num) const;
	std::string GenerateReturnType() const;

	bool IsConstructor() const;
	bool IsPureVirtual() const;
	bool NoReflection() const;
	bool NoSerialization() const;
};

struct Base
{
	AccessSpecifier access_specifier = AccessSpecifier::Public;
	std::string     name;

	bool is_virtual = false;
	bool is_packed  = false;
};

struct TypeMeta
{
	enum class Mode
	{
		Enum,
		Class,
		Struct
	};

	Mode                     mode = Mode::Class;
	std::vector<std::string> namespaces;
	std::vector<Parameter>   template_parameters;
	std::vector<Attribute>   attributes;
	std::string              name;
	std::vector<Base>        bases;
	std::vector<Field>       fields;

	bool        IsTemplateType() const;
	std::string GenerateName() const;
	std::string GenerateFullName() const;
	std::string GenerateTemplateList() const;
	bool        NeedReflection() const;
	bool        NeedSerialization() const;
	bool        NoSerialization() const;
	bool        NoReflection() const;
	bool        IsPureVirtual() const;
	bool        HasConstructor() const;
	bool        IsOverload(const std::string &name) const;
};
}        // namespace Meta
}        // namespace Ilum