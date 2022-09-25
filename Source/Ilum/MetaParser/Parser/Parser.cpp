#include "Parser.hpp"

namespace Ilum
{
MetaParser::~MetaParser()
{
	if (m_translation_unit)
		clang_disposeTranslationUnit(m_translation_unit);

	if (m_index)
		clang_disposeIndex(m_index);
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

	m_index = clang_createIndex(true, 1);
	std::string pre_include = "-I";
	std::string sys_include_temp;

	return false;
}

void MetaParser::GenerateFile()
{
}
}        // namespace Ilum