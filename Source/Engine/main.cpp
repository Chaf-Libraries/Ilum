#include "Engine.hpp"

#include <Core/Path.hpp>
#include <Core/Window.hpp>
#include <RHI/RHIContext.hpp>

#include <RenderCore/ShaderCompiler/ShaderCompiler.hpp>

using namespace Ilum;

int main()
{
	Ilum::Engine engine;

	engine.Tick();

	return 0;
}