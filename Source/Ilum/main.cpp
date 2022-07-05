#include "Application.hpp"

#include <Core/Path.hpp>

#include <sol/sol.hpp>

int main()
{
	sol::state lua;
	int        x = 0;
	lua.set_function("beep", [&x] { ++x; std::cout<<"Test"<<std::endl; });
	lua.script("beep()");
	assert(x == 1);

	Ilum::Path::GetInstance().SetCurrent(std::string(PROJECT_SOURCE_DIR));

	{
		Ilum::Application application;
		application.Tick();
	}

	return 0;
}