#include "Application.hpp"

#include <Core/Logger.hpp>
#include <Core/Timer.hpp>

#include <iostream>

int main()
{
	Ilum::Core::Logger::Initialize();

	Ilum::App::Application *application = new Ilum::App::Application;
	application->Run();
	delete application;

	Ilum::Core::Logger::Release();

#ifdef _WIN32
	_CrtDumpMemoryLeaks();
#endif

	return 0;
}