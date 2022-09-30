#include "Parser/Parser.hpp"

using namespace Ilum;

int main(int argc, const char *argv[])
{
	if (argc < 3)
	{
		return 0;
	}

	std::string              output_path = argv[1];
	std::vector<std::string> input_paths(argc - 2);
	for (int32_t i = 2; i < argc; i++)
	{
		input_paths[i - 2] = argv[i];
	}

	MetaParser parser(output_path, input_paths);
	parser.ParseProject();
	// parser.Parse();
	 parser.GenerateFile();

	return 0;
}