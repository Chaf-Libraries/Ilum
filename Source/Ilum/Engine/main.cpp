#include <Core/Hash.hpp>
#include <Core/Log.hpp>
#include <Core/Path.hpp>
#include <Core/Time.hpp>
#include <Core/Window.hpp>

#include <RHI/RHIContext.hpp>

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

		Ilum::ShaderDesc shader_desc = {};
		shader_desc.entry_point      = "main";
		shader_desc.stage            = Ilum::RHIShaderStage::Compute;
		shader_desc.source           = Ilum::ShaderSource::HLSL;
		shader_desc.target           = Ilum::ShaderTarget::DXIL;
		shader_desc.code             = "\
		RWTexture2D<float4> OutImage : register(u2);\
		struct CSParam\
		{\
			uint3 DispatchThreadID : SV_DispatchThreadID;\
		};\
		[numthreads(8, 8, 1)] void main(CSParam param) {\
			uint2 extent;\
			OutImage.GetDimensions(extent.x, extent.y);\
			float2 texel_size = 1.0 / float2(extent);\
			if (param.DispatchThreadID.x >= extent.x || param.DispatchThreadID.y >= extent.y)\
			{\
				return;\
			}\
    OutImage[param.DispatchThreadID.xy] = 1;\
		}\
";

		auto dxil          = Ilum::ShaderCompiler::GetInstance().Compile(shader_desc);
		shader_desc.target = Ilum::ShaderTarget::SPIRV;
		auto spirv         = Ilum::ShaderCompiler::GetInstance().Compile(shader_desc);

		Ilum::SpirvReflection::GetInstance().Reflect(spirv);


		while (window.Tick())
		{
			timer.Tick();

			context.BeginFrame();

			//if (i++ <= 3)
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