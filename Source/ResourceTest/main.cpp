#include <ResourceManager.hpp>
#include <RHI/RHIContext.hpp>
#include <Core/Window.hpp>

#include "_Generate/RHITexture.generate.hpp"
#include "_Generate/Resource.generate.hpp"
#include "_Generate/RHIDefinitions.generate.hpp"

#include <iostream>

using namespace Ilum;

int main()
{
	Window              window("Test", "", 100, 100);
	RHIContext      rhi_context(&window);
	{
		ResourceManager manager(&rhi_context);

		manager.Import<ResourceType::Texture>("Asset/Texture/Material/cavern-deposits/cavern-deposits_albedo.png");

		auto *resource = manager.GetResource<ResourceType::Texture>(15433844174579687259);

		manager.EraseResource<ResourceType::Texture>(15433844174579687259);
	}


	return 0;
}