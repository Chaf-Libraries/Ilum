#include "Parser.hpp"

namespace Ilum
{
MetaParser::MetaParser(const std::string &)
{
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

	m_index = clang_createIndex(true, 1);

	m_translation_unit = clang_createTranslationUnitFromSourceFile(
	    m_index, m_source_file_name.c_str(), static_cast<int32_t>(m_arguments.size()), m_arguments.data(), 0, nullptr);
	auto cursor = clang_getTranslationUnitCursor(m_translation_unit);

	//BuildClassAST(cursor)

	return false;
}

void MetaParser::GenerateFile()
{
}

bool MetaParser::ParseProject()
{


	return false;
}

void MetaParser::BuildClassAST()
{
}

std::string MetaParser::GetIncludeFile(const std::string &name)
{
	return std::string();
}
}        // namespace Ilum