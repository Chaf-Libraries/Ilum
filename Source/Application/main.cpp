#include "Application.hpp"

#include <Core/Logger.hpp>
#include <Core/Timer.hpp>

#include <Asset/Image2D.hpp>
#include <Asset/ImageCube.hpp>
#include <Asset/Mesh.hpp>

#include <iostream>

int main()
{
	//Ilum::Core::Logger::Initialize();

	{
		auto mesh = Ilum::Asset::Mesh::Create(std::string(PROJECT_SOURCE_DIR) + "Asset/Model/FlightHelmet/FlightHelmet.gltf");
	}


	//Ilum::App::Application *application = new Ilum::App::Application;
	//application->Run();
	//delete application;

	//Ilum::Core::Logger::Release();

#ifdef _WIN32
	_CrtDumpMemoryLeaks();
#endif

	return 0;
}