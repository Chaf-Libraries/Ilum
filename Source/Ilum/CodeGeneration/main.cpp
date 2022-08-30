#include <iostream>

#include <CPP14Lexer.h>
#include <CPP14Parser.h>
#include <CPP14ParserBaseListener.h>
#include <CPP14ParserBaseVisitor.h>
#include <antlr4-runtime.h>

#include "MetaGenerator.hpp"

using namespace antlr4;
using namespace Ilum;

int main(int argc, const char *argv[])
{
	std::ifstream stream;
	stream.open(argv[1]);
	ANTLRInputStream  input(stream);
	CPP14Lexer        lexer(&input);
	CommonTokenStream tokens(&lexer);
	CPP14Parser       parser(&tokens);
	tree::ParseTree  *tree = parser.translationUnit();
	TreeShapeVisitor  visitor;
	tree->accept(&visitor);

	return 0;
}