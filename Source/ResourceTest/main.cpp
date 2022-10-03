#include <ResourceManager.hpp>
#include <RHI/RHIContext.hpp>
#include <Core/Window.hpp>

#include <iostream>

using namespace Ilum;

int main()
{
	Window              window("Test", "", 100, 100);
	RHIContext      rhi_context(&window);
	ResourceManager     manager(&rhi_context);

	manager.Import<ResourceType::Texture2D>("Asset/Texture/node_editor_bg.png");

	auto *resource = manager.GetResource<ResourceType::Texture2D>(11657162107238754420);

	manager.EraseResource<ResourceType::Texture2D>(11657162107238754420);

	return 0;
}