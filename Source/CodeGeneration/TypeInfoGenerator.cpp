#include "TypeInfoGenerator.hpp"

#include <spdlog/fmt/fmt.h>

#include <set>
#include <sstream>

namespace Ilum
{
inline void ReflectAttribute(std::stringstream &sstream, const std::vector<Meta::Attribute> &attributes)
{
	if (!attributes.empty())
	{
		sstream << "(";
		for (uint32_t i = 0; i < attributes.size(); i++)
		{
			sstream << fmt::format("rttr::metadata(\"{}\", {})", attributes[i].name, attributes[i].value);
			if (i != attributes.size() - 1)
			{
				sstream << ", ";
			}
		}
		sstream << ")";
	}
}

inline void ReflectEnumeration(std::stringstream &sstream, const Meta::TypeMeta &meta)
{
	sstream << fmt::format("rttr::registration::enumeration<{}>(\"{}\")", meta.GenerateName(), meta.GenerateName());
	ReflectAttribute(sstream, meta.attributes);
	sstream << std::endl
	        << "(" << std::endl;
	for (size_t i = 0; i < meta.fields.size(); i++)
	{
		sstream << fmt::format("	rttr::value(\"{}\", {}::{})", meta.fields[i].name, meta.GenerateName(), meta.fields[i].name);
		if (i != meta.fields.size() - 1)
		{
			sstream << "," << std::endl;
		}
	}
	sstream << std::endl;
	sstream << ");" << std::endl
	        << std::endl;
}

inline void ReflectClass(std::stringstream &sstream, const Meta::TypeMeta &meta)
{
	if ((meta.IsPureVirtual() && !meta.HasConstructor()))
	{
		return;
	}

	sstream << fmt::format("rttr::registration::class_<{}>(\"{}\")", meta.GenerateName(), meta.GenerateName());
	ReflectAttribute(sstream, meta.attributes);
	for (auto &field : meta.fields)
	{
		if (field.name == "~" + meta.name ||
		    field.is_template ||
		    field.access_specifier != Meta::AccessSpecifier::Public ||
		    field.is_template ||
		    field.NoReflection())
		{
			// Deconstructor & template function
			continue;
		}

		// Function
		if (field.mode == Meta::Field::Mode::Function)
		{
			// Don't reflect operator overload
			if (field.name.substr(0, 8) == "operator")
			{
				continue;
			}

			// Constructor
			if (field.name == meta.name)
			{
				if (meta.IsPureVirtual())
				{
					continue;
				}
				sstream << std::endl
				        << "	.constructor";
				sstream << "<";
				for (uint32_t i = 0; i < field.parameters.size(); i++)
				{
					sstream << field.parameters[i].type;
					if (i != field.parameters.size() - 1)
					{
						sstream << ", ";
					}
				}
				sstream << ">()";
				ReflectAttribute(sstream, field.attributes);
			}
			else
			{
				if (field.IsConstructor())
				{
					sstream << std::endl
					        << "	.constructor(";
					sstream << fmt::format("&{}::{}",
					                       meta.GenerateName(),
					                       field.name);
					sstream << ")";
					ReflectAttribute(sstream, field.attributes);
				}
				else if (meta.IsOverload(field.name))
				{
					sstream << std::endl
					        << "	.method(";
					// static_cast<std::unique_ptr<RHIBuffer>(Ilum::RHIContext::*)(const BufferDesc &)>(&Ilum::RHIContext::CreateBuffer))
					sstream << fmt::format("\"{}\",  static_cast<{}({}::*)({})>(&{}::{})",
					                       field.name,
					                       field.GenerateReturnType(),
					                       meta.GenerateName(),
					                       field.GenerateParameterTypeList(),
					                       meta.GenerateName(),
					                       field.name);
					sstream << ")";
					ReflectAttribute(sstream, field.attributes);
				}
				else
				{
					sstream << std::endl
					        << "	.method(";
					sstream << fmt::format("\"{}\", &{}::{}",
					                       field.name,
					                       meta.GenerateName(),
					                       field.name);
					sstream << ")";
					ReflectAttribute(sstream, field.attributes);
				}
			}
		}
		// Variable
		else if (field.mode == Meta::Field::Mode::Variable)
		{
			sstream << std::endl
			        << "	.property(";
			sstream << fmt::format("\"{}\", &{}::{}",
			                       field.name,
			                       meta.GenerateName(),
			                       field.name);
			sstream << ")";
			ReflectAttribute(sstream, field.attributes);
		}
	}
	sstream << ";" << std::endl
	        << std::endl;
}

inline void SerializeClass(std::stringstream &sstream, const Meta::TypeMeta &meta)
{
	sstream << "template <class Archive>" << std::endl;
	sstream << fmt::format("void serialize(Archive& archive, {}& v)", meta.GenerateName()) << std::endl;
	sstream << "{" << std::endl;
	sstream << "	archive(";
	bool has_member = false;
	for (auto &field : meta.fields)
	{
		if (field.mode == Meta::Field::Mode::Variable &&
		    field.access_specifier == Meta::AccessSpecifier::Public)
		{
			sstream << "v." << field.name << ", ";
			has_member = true;
		}
	}
	if (has_member)
	{
		sstream.seekp(-2, std::ios_base::end);
	}
	sstream << ");" << std::endl;
	sstream << "}" << std::endl
	        << std::endl;
}

std::string GenerateTypeInfo(const std::vector<std::string> &headers, const std::vector<Meta::TypeMeta> &meta_types)
{
	std::stringstream result;

	result << "#pragma once" << std::endl
	       << "#include <rttr/registration.h>" << std::endl;
	for (auto &header : headers)
	{
		result << fmt::format("#include \"{}\"", header) << std::endl;
	}
	result << "// This code is generated by meta generator" << std::endl
	       << "namespace NAMESPACE_" << std::hash<uint64_t>()((uint64_t) &meta_types) << std::endl
	       << "{" << std::endl
	       << "RTTR_REGISTRATION" << std::endl
	       << "{" << std::endl
	       << "using namespace Ilum;" << std::endl
	       << std::endl;

	for (auto &meta : meta_types)
	{
		std::string namespace_ = "";
		for (auto &ns : meta.namespaces)
		{
			namespace_ += ns + "::";
		}

		std::string name = namespace_ + meta.name;

		switch (meta.mode)
		{
			case Meta::TypeMeta::Mode::Enum:
				ReflectEnumeration(result, meta);
				break;
			case Meta::TypeMeta::Mode::Class:
				if (meta.NeedReflection())
				{
					ReflectClass(result, meta);
				}
				break;
			case Meta::TypeMeta::Mode::Struct:
				ReflectClass(result, meta);
				break;
			default:
				break;
		}
	}

	result << "}" << std::endl;
	result << "}" << std::endl;
	result << std::endl
	       << "//Generate for Serialization" << std::endl;
	result << std::endl
	       << "namespace cereal" << std::endl
	       << "{" << std::endl;

	for (auto &meta : meta_types)
	{
		if (meta.mode == Meta::TypeMeta::Mode::Struct ||
		    (meta.mode == Meta::TypeMeta::Mode::Class && meta.NeedSerialization()))
		{
			SerializeClass(result, meta);
		}
	}

	result << std::endl
	       << "}" << std::endl;

	return result.str();
}
}        // namespace Ilum