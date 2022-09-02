#include "Engine.hpp"

#include <Core/Path.hpp>
#include <Core/Window.hpp>
#include <RHI/RHIContext.hpp>

#include <RenderCore/ShaderCompiler/ShaderCompiler.hpp>

using namespace Ilum;

int main()
{
	// Ilum::Window window("Ilum", "Asset/Icon/logo.bmp");
	// auto         context = Ilum::RHIContext(&window);
	// auto         bufferX = context.CreateBuffer<glm::vec2>(100, RHIBufferUsage::UnorderedAccess, RHIMemoryUsage::CPU_TO_GPU);
	// auto         bufferY = context.CreateBuffer<glm::vec2>(100, RHIBufferUsage::UnorderedAccess, RHIMemoryUsage::CPU_TO_GPU);
	// auto         texture = context.CreateTexture2D(100, 100, RHIFormat::R32G32B32A32_FLOAT, RHITextureUsage::UnorderedAccess, false);
	// auto         tex_size = context.CreateBuffer<glm::uvec2>(1, RHIBufferUsage::ConstantBuffer, RHIMemoryUsage::CPU_TO_GPU);
	// auto         time_buffer = context.CreateBuffer<float>(1, RHIBufferUsage::ConstantBuffer, RHIMemoryUsage::CPU_TO_GPU);

	// ShaderDesc           shader_desc = {};
	// std::vector<uint8_t> raw_data;
	// Path::GetInstance().Read("Source/Shaders/DrawUV.hlsl", raw_data);
	// shader_desc.code.resize(raw_data.size());
	// std::memcpy(shader_desc.code.data(), raw_data.data(), raw_data.size() * sizeof(uint8_t));
	// shader_desc.entry_point = "CSmain";
	// shader_desc.source      = ShaderSource::HLSL;
	// shader_desc.stage       = RHIShaderStage::Compute;
	// if (context.GetBackend() == RHIBackend::CUDA)
	//{
	//	shader_desc.target = ShaderTarget::PTX;
	// }
	// else if (context.GetBackend() == RHIBackend::Vulkan)
	//{
	//	shader_desc.target = ShaderTarget::SPIRV;
	// }
	// else if (context.GetBackend() == RHIBackend::DX12)
	//{
	//	shader_desc.target = ShaderTarget::DXIL;
	// }

	// ShaderMeta meta;

	// auto ptx = ShaderCompiler::GetInstance().Compile(shader_desc, meta);

	// auto shader         = context.CreateShader("CSmain", ptx);
	// auto descriptor     = context.CreateDescriptor(meta);
	// auto pipeline_state = context.CreatePipelineState();
	// pipeline_state->SetShader(RHIShaderStage::Compute, shader.get());
	// descriptor->BindBuffer("VarX", bufferX.get());
	////descriptor->BindBuffer("VarY", bufferY.get());
	////descriptor->BindTexture("Tex", texture.get(), RHITextureDimension::Texture2D);

	// uint32_t count = 0;
	// float time  = 0;
	// while (window.Tick())
	//{
	//	context.BeginFrame();

	//	time+=0.01f;

	//	time_buffer->Map();
	//	time_buffer->CopyToDevice(&time);
	//	time_buffer->Unmap();

	//	glm::uvec2 size = {context.GetBackBuffer()->GetDesc().width, context.GetBackBuffer()->GetDesc().height};
	//	tex_size->Map();
	//	tex_size->CopyToDevice(&size);
	//	tex_size->Unmap();

	//	descriptor->BindTexture("Tex", context.GetBackBuffer(), RHITextureDimension::Texture2D);
	//	descriptor->BindBuffer("TexSize", tex_size.get());
	//	descriptor->BindBuffer("Time", time_buffer.get());
	//	//TexSize
	//	glm::vec2 *dataX = static_cast<glm::vec2 *>(bufferX->Map());
	//	//glm::vec2 *dataY = static_cast<glm::vec2 *>(bufferY->Map());

	//	//LOG_INFO("({}, {}), ({}, {})", dataX[0].x, dataX[0].y, dataY[0].x, dataY[0].y);
	//	LOG_INFO("({}, {})", dataX[0].x, dataX[0].y);

	//	auto cmd_buffer = context.CreateCommand(RHIQueueFamily::Graphics);
	//	cmd_buffer->Begin();
	//	cmd_buffer->ResourceStateTransition({TextureStateTransition{
	//	                                        context.GetBackBuffer(),
	//	                                        RHIResourceState::Present,
	//	                                        RHIResourceState::UnorderedAccess,
	//	                                        TextureRange{
	//	                                            RHITextureDimension::Texture2D,
	//	                                            0, 1, 0, 1}}},
	//	                                    {});
	//	cmd_buffer->BindDescriptor(descriptor.get());
	//	cmd_buffer->BindPipelineState(pipeline_state.get());
	//	cmd_buffer->Dispatch(context.GetBackBuffer()->GetDesc().width, context.GetBackBuffer()->GetDesc().height, 1, 8, 8, 1);
	//	cmd_buffer->ResourceStateTransition({TextureStateTransition{
	//	                                        context.GetBackBuffer(),
	//	                                        RHIResourceState::UnorderedAccess,
	//	                                        RHIResourceState::Present,
	//	                                        TextureRange{
	//	                                            RHITextureDimension::Texture2D,
	//	                                            0, 1, 0, 1}}},
	//	                                    {});
	//	cmd_buffer->End();

	//	context.GetQueue(RHIQueueFamily::Graphics)->Submit({cmd_buffer});

	//	//std::this_thread::sleep_for(std::chrono::milliseconds(16));

	//	context.EndFrame();
	//}

	Ilum::Engine engine;

	engine.Tick();

	return 0;
}