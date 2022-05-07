#include <Core/Delegates.hpp>
#include <Core/Input.hpp>
#include <Core/JobSystem.hpp>
#include <Core/Macro.hpp>
#include <Core/Path.hpp>
#include <Core/Time.hpp>

#include <RHI/Buffer.hpp>
#include <RHI/Command.hpp>
#include <RHI/DescriptorState.hpp>
#include <RHI/Device.hpp>
#include <RHI/FrameBuffer.hpp>
#include <RHI/ImGuiContext.hpp>
#include <RHI/PipelineState.hpp>
#include <RHI/Texture.hpp>

#include <Render/RGBuilder.hpp>
#include <Render/RenderGraph.hpp>
#include <Render/RenderPass.hpp>
#include <Render/Renderer.hpp>

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
	std::function<void()> test;

	{
		int a = 100;
		test  = [=]() { std::cout << a; };
	}

	test();

	Path::GetInstance().SetCurrent(std::string(PROJECT_SOURCE_DIR));

	Window window("Ilum", "Asset/Icon/logo.bmp", 500, 500);

	Input::GetInstance().Bind(&window);

	RHIDevice device(&window);

	Ilum::ImGuiContext imgui_context(&window, &device);

	Renderer renderer(&device);

	uint32_t count = 0;

	TextureDesc rt_desc = {};
	rt_desc.width       = 500;
	rt_desc.height      = 500;
	rt_desc.depth       = 1;
	rt_desc.mips        = 1;
	rt_desc.layers      = 1;
	rt_desc.format      = VK_FORMAT_R8G8B8A8_UNORM;
	rt_desc.usage       = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

	TextureViewDesc view_desc  = {};
	view_desc.aspect           = VK_IMAGE_ASPECT_COLOR_BIT;
	view_desc.view_type        = VK_IMAGE_VIEW_TYPE_2D;
	view_desc.base_array_layer = 0;
	view_desc.base_mip_level   = 0;
	view_desc.layer_count      = 1;
	view_desc.level_count      = 1;

	ShaderDesc vertex_shader  = {};
	vertex_shader.filename    = "./Source/Shaders/triangle.vert";
	vertex_shader.entry_point = "main";
	vertex_shader.stage       = VK_SHADER_STAGE_VERTEX_BIT;
	vertex_shader.type        = ShaderType::GLSL;

	ShaderDesc frag_shader  = {};
	frag_shader.filename    = "./Source/Shaders/triangle.frag";
	frag_shader.entry_point = "main";
	frag_shader.stage       = VK_SHADER_STAGE_FRAGMENT_BIT;
	frag_shader.type        = ShaderType::GLSL;

	Texture render_target(&device, rt_desc);
	Sampler sampler(&device, SamplerDesc{});

	FrameBuffer framebuffer;
	framebuffer.Bind(&render_target, view_desc, ColorAttachmentInfo{});

	PipelineState pso;

	DynamicState dynamic_state = {};
	dynamic_state.dynamic_states.push_back(VK_DYNAMIC_STATE_SCISSOR);
	dynamic_state.dynamic_states.push_back(VK_DYNAMIC_STATE_VIEWPORT);

	ColorBlendState color_blend_state = {};
	color_blend_state.attachment_states.push_back(ColorBlendAttachmentState{});

	pso.SetDynamicState(dynamic_state);
	pso.SetColorBlendState(color_blend_state);

	pso.LoadShader(vertex_shader);
	pso.LoadShader(frag_shader);

	while (window.Tick())
	{
		Timer::GetInstance().Tick();
		device.NewFrame();

		float width  = 0;
		float height = 0;

		// Update ImGui
		{
			imgui_context.BeginFrame();

			renderer.OnImGui(imgui_context);

			imgui_context.EndFrame();
		}

		renderer.Tick();

		// Bake command buffer
		{
			auto &cmd_buffer = device.RequestCommandBuffer();
			cmd_buffer.Begin();
			// Render UI
			imgui_context.Render(cmd_buffer);
			cmd_buffer.End();

			device.Submit(cmd_buffer);
		}

		device.EndFrame();

		if (count++ > 100)
		{
			LOG_INFO("{}", Timer::GetInstance().FrameRate());
			count = 0;
		}
	}

	return 0;
}