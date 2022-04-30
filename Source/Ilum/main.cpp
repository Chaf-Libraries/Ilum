#include <Core/Delegates.hpp>
#include <Core/JobSystem.hpp>
#include <Core/Macro.hpp>
#include <Core/Time.hpp>
#include <Core/Window.hpp>
#include <Core/Input.hpp>

#include <RHI/Device.hpp>

#include <array>
#include <iostream>

void Fuck(int a, int b)
{
	LOG_INFO("Test {} {}", a, b);
}

using namespace Ilum;

std::atomic<uint32_t> count;

std::thread::id test_func()
{
	std::this_thread::sleep_for(std::chrono::milliseconds(10));
	return std::this_thread::get_id();
}

int main()
{
	Window window("Ilum", "Test", 500, 500);

	window.OnKeyFunc += [](int32_t key, int32_t scancode, int32_t action, int32_t mods) {
		LOG_INFO("key {}, scancode {}, action {}, mods {}", key, scancode, action, mods);
	};

	Input::GetInstance().Bind(&window);

	RHIDevice device(&window);

	while (window.Tick())
	{
		std::this_thread::sleep_for(std::chrono::milliseconds(16));
		LOG_INFO("{}", Input::GetInstance().IsKeyPressed(KEY_W));
	}

	return 0;
}