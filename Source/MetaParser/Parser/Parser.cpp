#include "Parser.hpp"
#include "Generator/ReflectionGenerator.hpp"
#include "Type/Class.hpp"
#include "Type/Enum.hpp"
#include "Type/Field.hpp"

#include <filesystem>
#include <fstream>

namespace Ilum
{
MetaParser::MetaParser(const std::string &output_path, const std::vector<std::string> &input_paths) :
    m_output_path(output_path), m_input_paths(input_paths)
{
	m_generators.emplace_back(std::make_unique<ReflectionGenerator>());
	std::filesystem::current_path(PROJECT_SOURCE_DIR);

	std::ofstream os(std::string(PROJECT_SOURCE_DIR) + "/bin/meta_headers.h");
	for (auto &input : m_input_paths)
	{
		os << "#include \"" + input + "\"" << std::endl;
	}
	os.flush();
	os.close();
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

bool MetaParser::Parse(const std::string &input_path)
{
	m_index = clang_createIndex(true, 0);

	std::string include_path = std::string("-I") + PROJECT_SOURCE_DIR + "/Source/Ilum";
	m_arguments.push_back(include_path.c_str());
	m_translation_unit = clang_createTranslationUnitFromSourceFile(
	    m_index, input_path.c_str(), static_cast<int32_t>(m_arguments.size()), m_arguments.data(), 0, nullptr);
	auto cursor = clang_getTranslationUnitCursor(m_translation_unit);

	Namespace current_namespace;
	BuildClassAST(cursor, current_namespace);
	current_namespace.clear();

	return true;
}

void MetaParser::GenerateFile()
{
	std::string output_dir;
	size_t      last_index = m_output_path.find_last_of("\\/");

	if (last_index != std::string::npos)
	{
		output_dir = m_output_path.substr(0, last_index + 1) + "_Generate/";
	}

	std::filesystem::create_directories(output_dir);

	for (auto &generator : m_generators)
	{
		for (auto &[path, schema] : m_schema_modules)
		{
			if (path.empty())
			{
				continue;
			}

			std::string output_path;
			auto        filename = std::filesystem::u8path(path).filename().generic_string();

			size_t last_index = filename.find_last_of('.');

			if (last_index != std::string::npos)
			{
				output_path = output_dir + filename.substr(0, last_index) + ".generate.hpp";
			}


			generator->Generate(path, output_path, schema);
		}
		generator->OutputFile(m_output_path);
	}
}

bool MetaParser::ParseProject()
{
	Parse(std::string(PROJECT_SOURCE_DIR) + "/bin/meta_headers.h");

	return true;
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
				auto class_ptr = std::make_shared<Class>(this, child, current_namespace);
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