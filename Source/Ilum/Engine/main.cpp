#include <Core/Hash.hpp>
#include <Core/Log.hpp>
#include <Core/Time.hpp>
#include <Core/Window.hpp>

#include <RHI/RHIContext.hpp>

int main()
{
	{
		Ilum::Window window("Ilum", "Asset/Icon/logo.bmp");

		auto device  = Ilum::RHIDevice::Create();
		auto texture = Ilum::RHITexture::Create(device.get(), Ilum::TextureDesc{10, 10, 1, 1, 1, 1, Ilum::RHIFormat::R8G8B8A8_UNORM, Ilum::RHITextureUsage::ShaderResource});

		/*
		    uint32_t width;
	uint32_t height;
	uint32_t depth;
	uint32_t mips;
	uint32_t layers;
	uint32_t samples;

	RHIFormat           format;
	RHITextureUsage     usage;
		*/

		/*Ilum::RHIContext context(&window);

		Ilum::Timer timer;

		uint32_t i = 0;

		while (window.Tick())
		{
		    timer.Tick();
		    context.BeginFrame();

		    auto *cmd = context.CreateCommand(Ilum::RHIQueueFamily::Graphics);
		    cmd->Begin();
		    if (i++ <= 3)
		    {
		        cmd->ResourceStateTransition({Ilum::TextureStateTransition{
		                                         context.GetBackBuffer(),
		                                         Ilum::RHITextureState::Undefined,
		                                         Ilum::RHITextureState::Present,
		                                         Ilum::TextureRange{
		                                             Ilum::RHITextureDimension::Texture2D,
		                                             0, 1, 0, 1}}},
		                                     {});
		    }
		    cmd->End();
		    context.GetQueue(Ilum::RHIQueueFamily::Graphics)->Submit({cmd});
		    window.SetTitle(fmt::format("IlumEngine FPS: {}", timer.FrameRate()));
		    context.EndFrame();
		}*/
	}

	return 0;
}