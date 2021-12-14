#include <Ilum/Device/Window.hpp>
#include <Ilum/Editor/Editor.hpp>
#include <Ilum/Engine/Context.hpp>
#include <Ilum/Engine/Engine.hpp>
#include <Ilum/Scene/Scene.hpp>
#include <Ilum/Scene/System.hpp>
#include <Ilum/Timing/Timer.hpp>

int main()
{
	Ilum::Engine engine;

	Ilum::Window::instance()->setIcon(std::string(PROJECT_SOURCE_DIR) + "Asset/Texture/Icon/logo.bmp");

	while (!Ilum::Window::instance()->shouldClose())
	{
		engine.onTick();

		Ilum::Window::instance()->setTitle((Ilum::Scene::instance()->name.empty() ? "IlumEngine" : Ilum::Scene::instance()->name) + " FPS: " + std::to_string(Ilum::Timer::instance()->getFPS()));
	}

	return 0;
}