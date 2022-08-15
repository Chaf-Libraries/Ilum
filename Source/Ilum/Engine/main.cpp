#include <Core/Hash.hpp>
#include <Core/Log.hpp>
#include <Core/Path.hpp>
#include <Core/Time.hpp>
#include <Core/Window.hpp>

#include <RHI/RHIContext.hpp>
#include <RHI/RHIDescriptor.hpp>

#include <RenderCore/ShaderCompiler/ShaderCompiler.hpp>
#include <RenderCore/ShaderCompiler/SpirvReflection.hpp>

int main()
{
	{
		Ilum::Window window("Ilum", "Asset/Icon/logo.bmp");

		Ilum::RHIContext context(&window);

		Ilum::Timer timer;
		Ilum::Timer stopwatch;
		auto        current = Ilum::Path::GetInstance().GetCurrent();

		uint32_t i = 0;

		std::vector<uint8_t> shader_data;
		Ilum::Path::GetInstance().Read("E:/Workspace/Ilum/Source/Shaders/DrawUV.hlsl", shader_data, false);
		// Ilum::Path::GetInstance().Read("E:/Workspace/Vulkan/data/shaders/hlsl/computeraytracing/raytracing.comp", shader_data, false);

		std::string shader_data_string;
		shader_data_string.resize(shader_data.size());
		std::memcpy(shader_data_string.data(), shader_data.data(), shader_data.size());

		Ilum::ShaderDesc vertex_shader_desc = {};
		vertex_shader_desc.entry_point      = "VSmain";
		vertex_shader_desc.stage            = Ilum::RHIShaderStage::Vertex;
		vertex_shader_desc.source           = Ilum::ShaderSource::HLSL;
		vertex_shader_desc.target           = Ilum::ShaderTarget::SPIRV;
		vertex_shader_desc.code             = shader_data_string;

		Ilum::ShaderDesc pixel_shader_desc = {};
		pixel_shader_desc.entry_point      = "PSmain";
		pixel_shader_desc.stage            = Ilum::RHIShaderStage::Fragment;
		pixel_shader_desc.source           = Ilum::ShaderSource::HLSL;
		pixel_shader_desc.target           = Ilum::ShaderTarget::SPIRV;
		pixel_shader_desc.code             = shader_data_string;

		Ilum::ShaderMeta vertex_meta = {};
		Ilum::ShaderMeta pixel_meta  = {};

		auto vertex_shader_spirv = Ilum::ShaderCompiler::GetInstance().Compile(vertex_shader_desc, vertex_meta);
		auto pixel_shader_spirv  = Ilum::ShaderCompiler::GetInstance().Compile(pixel_shader_desc, pixel_meta);

		auto vertex_shader = context.CreateShader("VSmain", vertex_shader_spirv);
		auto pixel_shader  = context.CreateShader("PSmain", pixel_shader_spirv);

		Ilum::ShaderMeta meta = vertex_meta;
		meta += pixel_meta;

		auto descriptor = context.CreateDescriptor(meta);
		auto pso        = context.CreatePipelineState();

		Ilum::DepthStencilState depth_stencil_state;
		depth_stencil_state.depth_test_enable = false;
		depth_stencil_state.depth_write_enable = false;

		Ilum::BlendState blend_state;
		blend_state.attachment_states.resize(1);

		Ilum::RasterizationState rasterization_state;
		rasterization_state.cull_mode = Ilum::RHICullMode::None;

		pso->SetDepthStencilState(depth_stencil_state);
		pso->SetBlendState(blend_state);
		pso->SetRasterizationState(rasterization_state);

		auto result = context.CreateTexture2D(100, 100, Ilum::RHIFormat::R16G16B16A16_FLOAT, Ilum::RHITextureUsage::RenderTarget | Ilum::RHITextureUsage::ShaderResource, false);

		auto render_target = context.CreateRenderTarget();
		Ilum::ColorAttachment attachment    = {};
		//attachment.clear_value[0]           = 1.f;
		//attachment.clear_value[3]           = 1.f;
		render_target->Add(result.get(), Ilum::RHITextureDimension::Texture2D, attachment);

		// auto texture2 = context.CreateTexture2D(100, 100, Ilum::RHIFormat::R8G8B8A8_UNORM, Ilum::RHITextureUsage::ShaderResource | Ilum::RHITextureUsage::UnorderedAccess, false);
		// descriptor->BindTexture("Result", result.get(), Ilum::RHITextureDimension::Texture2D);

		pso->SetShader(Ilum::RHIShaderStage::Vertex, vertex_shader.get());
		pso->SetShader(Ilum::RHIShaderStage::Fragment, pixel_shader.get());

		// descriptor->BindTexture("InImage", {texture1.get()}, Ilum::RHITextureDimension::Texture2D);
		// descriptor->BindTexture("OutImage", {texture2.get()}, Ilum::RHITextureDimension::Texture2D);

		while (window.Tick())
		{
			timer.Tick();

			context.BeginFrame();

			// if (i++ <= 3)
			{
				auto *cmd = context.CreateCommand(Ilum::RHIQueueFamily::Graphics);
				cmd->Begin();

				cmd->ResourceStateTransition({Ilum::TextureStateTransition{
				                                 result.get(),
				                                 Ilum::RHITextureState::Undefined,
				                                 Ilum::RHITextureState::RenderTarget,
				                                 Ilum::TextureRange{
				                                     Ilum::RHITextureDimension::Texture2D,
				                                     0, 1, 0, 1}}},
				                             {});

				cmd->BindDescriptor(descriptor.get());
				cmd->BeginRenderPass(render_target.get());
				cmd->BindPipelineState(pso.get());
				cmd->SetViewport(100, 100);
				cmd->SetScissor(100, 100);
				cmd->Draw(3, 1, 0, 0);
				cmd->EndRenderPass();
				cmd->ResourceStateTransition({Ilum::TextureStateTransition{
				                                 context.GetBackBuffer(),
				                                 Ilum::RHITextureState::Undefined,
				                                 Ilum::RHITextureState::Present,
				                                 Ilum::TextureRange{
				                                     Ilum::RHITextureDimension::Texture2D,
				                                     0, 1, 0, 1}}},
				                             {});
				cmd->End();
				context.GetQueue(Ilum::RHIQueueFamily::Graphics)->Submit({cmd});
			}

			window.SetTitle(fmt::format("IlumEngine FPS: {}", timer.FrameRate()));
			context.EndFrame();
		}
	}

	return 0;
}