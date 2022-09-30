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
	MetaParser(const std::string &output_path, const std::vector<std::string> &input_paths);

	~MetaParser();

	void Finish();

	bool Parse(const std::string &input_path);

	bool ParseProject();

	void GenerateFile();

  private:
	void        BuildClassAST(const Cursor &cursor, Namespace &current_namespace);
	std::string GetIncludeFile(const std::string &name);

  private:
	std::string              m_output_path;
	std::vector<std::string> m_input_paths;

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