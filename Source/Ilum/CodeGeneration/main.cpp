#include <iostream>

#include <CPP14Lexer.h>
#include <CPP14Parser.h>
#include <CPP14ParserBaseListener.h>
#include <CPP14ParserBaseVisitor.h>
#include <antlr4-runtime.h>

#include "MetaGenerator.hpp"
#include "TypeInfoGenerator.hpp"

#include <filesystem>

using namespace antlr4;
using namespace Ilum;

int main(int argc, const char *argv[])
{
	std::vector<Meta::MetaType> meta_types;
	std::vector<std::string>    headers;

	for (int32_t i = 2; i < argc; i++)
	{
		std::cout << "Generate meta info from " << argv[i] << std::endl;

		std::ifstream stream;
		stream.open(argv[i]);
		ANTLRInputStream  input(stream);
		CPP14Lexer        lexer(&input);
		CommonTokenStream tokens(&lexer);
		CPP14Parser       parser(&tokens);
		tree::ParseTree  *tree = parser.translationUnit();
		TreeShapeVisitor  visitor;
		tree->accept(&visitor);

		headers.push_back(argv[i]);

		auto &meta_type_ = visitor.GetMetaTypes();
		meta_types.insert(meta_types.end(), std::move_iterator(meta_type_.begin()), std::move_iterator(meta_type_.end()));
	}

	std::string   result = GenerateTypeInfo(headers, meta_types);
	std::ofstream output(argv[1]);
	output.write(result.data(), result.size());
	output.flush();
	output.close();
	return 0;
}