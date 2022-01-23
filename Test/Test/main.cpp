#include <Core/Logger.hpp>
#include <Core/Window.hpp>

int main()
{
	{
		Ilum::Core::Logger::Initialize();

		LOG_INFO("Test {}", "info");
		LOG_TRACE("Test {}", "trace");
		LOG_WARN("Test {}", "warn");
		LOG_ERROR("Test {}", "error");
		LOG_CRITICAL("Test {}", "critical");

		Ilum::Core::WindowDesc desc;
		desc.backend = Ilum::Core::GraphicsBackend::OpenGL;
		desc.title   = "IlumEngine";
		desc.width   = 1000;
		desc.height  = 1000;
		auto* window = Ilum::Core::Window::Create(desc);

		while (true)
		{
			window->OnUpdate();
		}

		delete window;
		Ilum::Core::Logger::Release();
	}

	return 0;
}