#include <Core/Hash.hpp>
#include <Core/Log.hpp>
#include <Core/Window.hpp>
#include <Core/Time.hpp>

#include <RHI/RHIContext.hpp>

int main()
{
	{
		Ilum::Window window("Ilum", "Asset/Icon/logo.bmp");
		Ilum::RHIContext context(&window);

		Ilum::Timer timer;

		while (window.Tick())
		{
			timer.Tick();
			context.BeginFrame();

			context.EndFrame();
		}
	}

	return 0;
}