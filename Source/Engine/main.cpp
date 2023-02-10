#include "Engine.hpp"

#include <ShaderCompiler/ShaderCompiler.hpp> 

int main()
{
	Ilum::Engine engine;
	//Ilum::ShaderCompiler::GetInstance();

	engine.Tick();

	return 0;
}