#include <Core/Delegates.hpp>
#include <Core/JobSystem.hpp>
#include <Core/Macro.hpp>
#include <Core/Time.hpp>
#include <Core/Window.hpp>
#include <Core/Input.hpp>

#include <RHI/Device.hpp>
#include <RHI/Texture.hpp>
#include <RHI/Buffer.hpp>

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

	TextureDesc desc;
	desc.width = 100;
	desc.height = 100;
	desc.format = VK_FORMAT_R8G8B8A8_UNORM;
	desc.usage  = VK_IMAGE_USAGE_SAMPLED_BIT;

	BufferDesc buffer_desc = {};
	buffer_desc.buffer_usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
	buffer_desc.memory_usage = VMA_MEMORY_USAGE_GPU_ONLY;
	buffer_desc.size         = 100;

	auto texture = device.CreateTexture(desc);
	auto buffer  = device.CreateBuffer(buffer_desc);

	while (window.Tick())
	{
		std::this_thread::sleep_for(std::chrono::milliseconds(16));
		LOG_INFO("{}", Input::GetInstance().IsKeyPressed(KEY_W));
	}

	return 0;
}