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
		Ilum::Path::GetInstance().Read("D:/Workspace/Vulkan/data/shaders/hlsl/instancing/instancing.vert", shader_data);

		std::string shader_data_string;
		shader_data_string.resize(shader_data.size());
		std::memcpy(shader_data_string.data(), shader_data.data(), shader_data.size());

		Ilum::ShaderDesc shader_desc = {};
		shader_desc.entry_point      = "main";
		shader_desc.stage            = Ilum::RHIShaderStage::Vertex;
		shader_desc.source           = Ilum::ShaderSource::HLSL;
		shader_desc.target           = Ilum::ShaderTarget::DXIL;
		shader_desc.code             = shader_data_string;

		Ilum::ShaderMeta meta = {};
		auto             dxil = Ilum::ShaderCompiler::GetInstance().Compile(shader_desc, meta);
		shader_desc.target    = Ilum::ShaderTarget::SPIRV;
		auto spirv            = Ilum::ShaderCompiler::GetInstance().Compile(shader_desc, meta);

		auto descriptor = context.CreateDescriptor(meta);

		//auto texture1 = context.CreateTexture2D(100, 100, Ilum::RHIFormat::R8G8B8A8_UNORM, Ilum::RHITextureUsage::ShaderResource | Ilum::RHITextureUsage::UnorderedAccess, false);
		//auto texture2 = context.CreateTexture2D(100, 100, Ilum::RHIFormat::R8G8B8A8_UNORM, Ilum::RHITextureUsage::ShaderResource | Ilum::RHITextureUsage::UnorderedAccess, false);

		//descriptor->BindTexture("InImage", {texture1.get()}, Ilum::RHITextureDimension::Texture2D);
		//descriptor->BindTexture("OutImage", {texture2.get()}, Ilum::RHITextureDimension::Texture2D);

		while (window.Tick())
		{
			timer.Tick();

			context.BeginFrame();

			// if (i++ <= 3)
			{
				auto *cmd = context.CreateCommand(Ilum::RHIQueueFamily::Graphics);
				cmd->Begin();
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
			// LOG_INFO("End Frame: {} ms", stopwatch.DeltaTime());
		}
	}

	return 0;
}