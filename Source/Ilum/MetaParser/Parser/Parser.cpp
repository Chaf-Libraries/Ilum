#include "Parser.hpp"
#include "Generator/ReflectionGenerator.hpp"
#include "Type/Class.hpp"
#include "Type/Enum.hpp"
#include "Type/Field.hpp"

namespace Ilum
{
MetaParser::MetaParser(const std::string &file_path) :
    m_source_file_name(file_path)
{
	m_generators.emplace_back(std::make_unique<ReflectionGenerator>());
}

MetaParser::~MetaParser()
{
	if (m_translation_unit)
	{
		clang_disposeTranslationUnit(m_translation_unit);
	}

	if (m_index)
	{
		clang_disposeIndex(m_index);
	}
}

void MetaParser::Finish()
{
}

bool MetaParser::Parse()
{
	if (ParseProject())
	{
		std::cerr << "Parsing project file error!" << std::endl;
		return false;
	}

	m_index = clang_createIndex(true, 0);

	m_translation_unit = clang_createTranslationUnitFromSourceFile(
	    m_index, m_source_file_name.c_str(), static_cast<int32_t>(m_arguments.size()), m_arguments.data(), 0, nullptr);
	auto cursor = clang_getTranslationUnitCursor(m_translation_unit);

	Namespace current_namespace;
	BuildClassAST(cursor, current_namespace);
	current_namespace.clear();

	return true;
}

void MetaParser::GenerateFile()
{
	for (auto& generator : m_generators)
	{
		for (auto& [path, schema] : m_schema_modules)
		{
			generator->Generate(path, schema);
		}
		generator->Finish();
	}
}

bool MetaParser::ParseProject()
{
	return false;
}

void MetaParser::BuildClassAST(const Cursor &cursor, Namespace &current_namespace)
{
	for (auto &child : cursor.GetChildren())
	{
		auto kind = child.GetKind();

		if (child.IsDefinition() && (kind == CXCursor_ClassDecl || kind == CXCursor_StructDecl || kind == CXCursor_EnumDecl))
		{
			if (kind == CXCursor_ClassDecl || kind == CXCursor_StructDecl)
			{
				auto class_ptr = std::make_shared<Class>(child, current_namespace);
				if (class_ptr->ShouldCompile())
				{
					auto file = class_ptr->GetSourceFile();
					m_schema_modules[file].classes.emplace_back(class_ptr);
					m_type_table[class_ptr->GetName()] = file;
				}
			}
			if (kind == CXCursor_EnumDecl)
			{
				auto enum_ptr = std::make_shared<Enum>(child, current_namespace);
				if (enum_ptr->ShouldCompile())
				{
					auto file = enum_ptr->GetSourceFile();
					m_schema_modules[file].enums.emplace_back(enum_ptr);
					m_type_table[enum_ptr->GetName()] = file;
				}
			}
		}
		else
		{
			if (kind == CXCursor_Namespace)
			{
				auto display_name = child.GetDisplayName();
				if (!display_name.empty())
				{
					current_namespace.emplace_back(display_name);
					BuildClassAST(child, current_namespace);
					current_namespace.pop_back();
				}
			}
		}
	}
}

std::string MetaParser::GetIncludeFile(const std::string &name)
{
	return std::string();
}
}        // namespace Ilum