#include "Application.hpp"

#include <Core/Path.hpp>

#include <sol/sol.hpp>

#include <Render/Material/MaterialNode/Constant.hpp>
#include <Render/Material/MaterialGraph.hpp>

namespace Ilum
{
	using fuck=int;
}

int main()
{

	Ilum::Path::GetInstance().SetCurrent(std::string(PROJECT_SOURCE_DIR));

	{
		Ilum::Application application;
		application.Tick();
	}

	return 0;
}