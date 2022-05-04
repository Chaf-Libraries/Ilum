#include <Core/Delegates.hpp>
#include <Core/JobSystem.hpp>
#include <Core/Macro.hpp>
#include <Core/Time.hpp>
#include <Core/Input.hpp>
#include <Core/Path.hpp>

#include <RHI/Device.hpp>
#include <RHI/Texture.hpp>
#include <RHI/Buffer.hpp>
#include <RHI/ImGuiContext.hpp>
#include <RHI/Command.hpp>

#include <Render/RenderGraph.hpp>

#include <array>
#include <iostream>

#include <imgui.h>

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
	Path::GetInstance().SetCurrent(std::string(PROJECT_SOURCE_DIR));

	Window window("Ilum", "Asset/Icon/logo.bmp", 500, 500);

	window.OnKeyFunc += [](int32_t key, int32_t scancode, int32_t action, int32_t mods) {
		LOG_INFO("key {}, scancode {}, action {}, mods {}", key, scancode, action, mods);
	};

	Input::GetInstance().Bind(&window);

	RHIDevice device(&window);

	RenderGraph rg(&device);

	Ilum::ImGuiContext imgui_context(&window, &device);

	uint32_t count = 0;

	while (window.Tick())
	{
		Timer::GetInstance().Tick();
		device.NewFrame();
		imgui_context.BeginFrame();
		
		ImGui::ShowDemoWindow();

		rg.OnImGui();

		imgui_context.EndFrame();

		auto& cmd_buffer=device.RequestCommandBuffer();

		cmd_buffer.Begin();
		imgui_context.Render(cmd_buffer);
		cmd_buffer.End();

		device.Submit(cmd_buffer);
		device.EndFrame();

		if (count++ > 100)
		{
			LOG_INFO("{}", Timer::GetInstance().FrameRate());
			count = 0;
		}
	}

	return 0;
}