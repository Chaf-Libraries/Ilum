#include "Application.hpp"

#include <Core/Path.hpp>

int main()
{
	Ilum::Path::GetInstance().SetCurrent(std::string(PROJECT_SOURCE_DIR));

	{
		Ilum::Application application;
		application.Tick();
	}

	return 0;
}