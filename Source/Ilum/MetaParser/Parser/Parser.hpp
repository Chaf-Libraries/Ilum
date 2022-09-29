#pragma once

#include "Generator/Generator.hpp"
#include "Meta/MetaInfo.hpp"
#include "Meta/MetaUtils.hpp"
#include "Precompile.hpp"
#include "Type/SchemaModule.hpp"

namespace Ilum
{
class MetaParser
{
  public:
	MetaParser(const std::string &file_path);

	~MetaParser();

	void Finish();

	bool Parse();

	void GenerateFile();

  private:
	bool        ParseProject();
	void        BuildClassAST(const Cursor &cursor, Namespace &current_namespace);
	std::string GetIncludeFile(const std::string &name);

  private:
	std::vector<std::string> m_work_paths;

	std::string m_source_file_name;

	CXIndex           m_index            = {};
	CXTranslationUnit m_translation_unit = {};

	std::unordered_map<std::string, std::string>  m_type_table;
	std::unordered_map<std::string, SchemaModule> m_schema_modules;

	std::vector<std::unique_ptr<Generator>> m_generators;

	std::vector<const char *> m_arguments = {{"-x",
	                                          "c++",
	                                          "-std=c++17",
	                                          "-D__REFLECTION_PARSER__",
	                                          "-DNDEBUG",
	                                          "-D__clang__",
	                                          "-w",
	                                          "-MG",
	                                          "-M",
	                                          "-ferror-limit=0",
	                                          "-o clangLog.txt"}};
};
}        // namespace Ilum