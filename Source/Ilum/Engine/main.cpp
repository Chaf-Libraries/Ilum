#include <Core/Hash.hpp>
#include <Core/Log.hpp>
#include <Core/Time.hpp>
#include <Core/Window.hpp>

#include <RHI/RHIContext.hpp>

int main()
{
	{
		Ilum::Window window("Ilum", "Asset/Icon/logo.bmp");

		Ilum::RHIContext context(&window);

		Ilum::Timer timer;
		Ilum::Timer stopwatch;

		uint32_t i = 0;

		while (window.Tick())
		{
		    timer.Tick();

			stopwatch.Tick();
		    context.BeginFrame();
			stopwatch.Tick();
			//LOG_INFO("Begin Frame: {} ms", stopwatch.DeltaTime());
		    /*auto *cmd = context.CreateCommand(Ilum::RHIQueueFamily::Graphics);
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
		    cmd->End();*/
		    //context.GetQueue(Ilum::RHIQueueFamily::Graphics)->Submit({cmd});
		    window.SetTitle(fmt::format("IlumEngine FPS: {}", timer.FrameRate()));
			stopwatch.Tick();
			context.EndFrame();
			stopwatch.Tick();
			//LOG_INFO("End Frame: {} ms", stopwatch.DeltaTime());
		}
	}

	return 0;
}