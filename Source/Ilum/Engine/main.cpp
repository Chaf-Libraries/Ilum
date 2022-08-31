#include "Engine.hpp"

#include <Core/Window.hpp>
#include <Core/Path.hpp>
#include <RHI/RHIContext.hpp>

#include <RenderCore/ShaderCompiler/ShaderCompiler.hpp>

using namespace Ilum;

int main()
{
	Ilum::Window window("Ilum", "Asset/Icon/logo.bmp");
	auto context = Ilum::RHIContext(&window);
	auto buffer = context.CreateBuffer(100 * sizeof(glm::vec2), RHIBufferUsage::UnorderedAccess, RHIMemoryUsage::CPU_TO_GPU);

	ShaderDesc shader_desc = {};
	std::vector<uint8_t> raw_data;
	Path::GetInstance().Read("Source/Shaders/DrawUV.hlsl", raw_data);
	shader_desc.code.resize(raw_data.size());
	std::memcpy(shader_desc.code.data(), raw_data.data(), raw_data.size() * sizeof(uint8_t));
	shader_desc.entry_point = "CSmain";
	shader_desc.source      = ShaderSource::HLSL;
	shader_desc.stage       = RHIShaderStage::Compute;
	shader_desc.target      = ShaderTarget::PTX;

	ShaderMeta meta;
	auto ptx = ShaderCompiler::GetInstance().Compile(shader_desc, meta);

	auto shader         = context.CreateShader("CSmain", ptx);
	auto descriptor = context.CreateDescriptor(meta);
	auto pipeline_state = context.CreatePipelineState();
	pipeline_state->SetShader(RHIShaderStage::Compute, shader.get());
	descriptor->BindBuffer("VarX", buffer.get());


	while (window.Tick())
	{
		glm::vec2 *data = static_cast<glm::vec2 *>(buffer->Map());

		context.BeginFrame();

		auto cmd_buffer = context.CreateCommand(RHIQueueFamily::Graphics);
		cmd_buffer->Begin();
		cmd_buffer->BindDescriptor(descriptor.get());
		cmd_buffer->BindPipelineState(pipeline_state.get());
		cmd_buffer->Dispatch(100, 100, 1, 8, 8, 1);
		cmd_buffer->End();

		context.GetQueue(RHIQueueFamily::Graphics)->Submit({cmd_buffer});

		std::this_thread::sleep_for(std::chrono::milliseconds(16));

		context.EndFrame();
	}

	//Ilum::Engine engine;


	//engine.Tick();

	return 0;
}