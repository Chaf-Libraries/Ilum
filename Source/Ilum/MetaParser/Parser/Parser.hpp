#pragma once

#include "Precompile.hpp"

#include <clang-c/Index.h>

namespace Ilum
{
class MetaParser
{
  public:
	MetaParser(const std::string &);

	~MetaParser();

	void Finish();

	bool Parse();

	void GenerateFile();

  private:
	std::vector<std::string> m_work_paths;

	CXIndex           m_index;
	CXTranslationUnit m_translation_unit;

	std::unordered_map<std::string, std::string> m_type_table;

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

  private:
	bool ParseProject();
	void BuildClassAST();
	std::string GetIncludeFile(const std::string &name);
};
}        // namespace Ilum