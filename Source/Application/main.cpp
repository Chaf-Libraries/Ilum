#include "Application.hpp"

#include <Core/Allocator/LinearAllocator.hpp>
#include <Core/Allocator/StackAllocator.hpp>
#include <Core/Logger.hpp>
#include <Core/Memory.hpp>
#include <Core/Timer.hpp>

#include <iostream>

int main()
{
	//Ilum::Core::Logger::Initialize();

	//Ilum::App::Application *application = new Ilum::App::Application;
	//application->Run();
	//delete application;

	Ilum::Core::Timer timer;
	{
		timer.Tick();
		Ilum::Core::StackAllocator allocator(1000 * sizeof(uint32_t));
		for (uint32_t i = 0; i < 1000; i++)
		{
			uint32_t *tmp = (uint32_t *) allocator.Allocate(sizeof(uint32_t));
			allocator.Free(tmp);
		}
		std::cout << timer.Elapsed() << std::endl;

		timer.Tick();
		for (uint32_t i = 0; i < 1000; i++)
		{
			uint32_t *tmp = (uint32_t *) malloc(sizeof(uint32_t));
			free(tmp);
		}
		std::cout << timer.Elapsed() << std::endl;
	}

	//Ilum::Core::Logger::Release();

	_CrtDumpMemoryLeaks();

	return 0;
}