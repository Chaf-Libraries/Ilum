#include <Ilum/Engine/Engine.hpp>
#include <Ilum/Engine/Context.hpp>
#include <Ilum/Device/Window.hpp>
#include <Ilum/Editor/Editor.hpp>

int main()
{
	Ilum::Engine engine;

	while (!Ilum::Window::instance()->shouldClose())
	{
		engine.onTick();
	}

	return 0;
}