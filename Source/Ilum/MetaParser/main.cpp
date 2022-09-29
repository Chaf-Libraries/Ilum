#include "Parser/Parser.hpp"

using namespace Ilum;

int main(int argc, const char *argv[])
{
	//MetaParser parser("E:/Workspace/Ilum/Source/Ilum/RHI/RHITexture.hpp");
	MetaParser parser("E:/Workspace/Ilum/Source/Ilum/MetaParser/Test.hpp");
	parser.Parse();
	parser.GenerateFile();

	return 0;
}