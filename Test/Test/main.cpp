#include <Core/Device/Input.hpp>
#include <Core/Device/Window.hpp>
#include <Core/Engine/Context.hpp>
#include <Core/Engine/Engine.hpp>
#include <Core/Engine/Pooling/MemoryPool.hpp>
#include <Core/Engine/Timing/Timer.hpp>

#include <Core/Device/Instance.hpp>

int main()
{
	Ilum::Engine engine;

	auto *window = engine.getContext().getSubsystem<Ilum::Window>();
	auto *timer  = engine.getContext().getSubsystem<Ilum::Timer>();

	const std::string title  = window->getTitle();

	{
		Ilum::Instance instance;
	}

	while (!window->shouldClose())
	{
		engine.onTick();

		std::this_thread::sleep_for(std::chrono::duration<double,std::milli>(16));

		window->setTitle(title + " FPS: " + std::to_string(timer->getFPS()));
	}

	return 0;
}