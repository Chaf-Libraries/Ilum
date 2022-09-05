#include <iostream>

#pragma warning(push, 0)
#include <CPP14Lexer.h>
#include <CPP14Parser.h>
#include <CPP14ParserBaseListener.h>
#include <CPP14ParserBaseVisitor.h>
#include <antlr4-runtime.h>
#pragma warning(pop)

#include "MetaGenerator.hpp"
#include "TypeInfoGenerator.hpp"

#include <filesystem>

using namespace antlr4;
using namespace Ilum;

int main(int argc, const char *argv[])
{
	std::vector<Meta::TypeMeta> meta_types;
	std::vector<std::string>    headers;

	for (int32_t i = 2; i < argc; i++)
	{
		std::cout << "Generate meta info from " << argv[i] << std::endl;

		std::ifstream stream;
		stream.open(argv[i]);

		stream.seekg(0, std::ios::end);
		uint64_t read_count = static_cast<uint64_t>(stream.tellg());
		stream.seekg(0, std::ios::beg);

		std::string code = "";
		code.resize(static_cast<size_t>(read_count));
		stream.read(reinterpret_cast<char *>(code.data()), read_count);

		stream.clear();
		stream.seekg(0);

		ANTLRInputStream  input(stream);
		CPP14Lexer        lexer(&input);
		CommonTokenStream tokens(&lexer);
		CPP14Parser       parser(&tokens);
		tree::ParseTree  *tree = parser.translationUnit();
		TreeShapeVisitor  visitor(code);
		//CPP14ParserBaseVisitor  visitor;
		tree->accept(&visitor);

		headers.push_back(argv[i]);

		auto &meta_type_ = visitor.GetTypeMeta();
		meta_types.insert(meta_types.end(), std::move_iterator(meta_type_.begin()), std::move_iterator(meta_type_.end()));

		stream.close();
	}

	std::string   result = GenerateTypeInfo(headers, meta_types);
	std::ofstream output(argv[1]);
	output.write(result.data(), result.size());
	output.flush();
	output.close();
	return 0;
}