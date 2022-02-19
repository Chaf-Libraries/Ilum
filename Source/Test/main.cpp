#include <Resource/ResourceCache.hpp>
#include <Core/JobSystem/JobSystem.hpp>

int main()
{
	Ilum::Core::JobSystem::Initialize();

	auto &image1 = Ilum::Resource::ResourceCache::LoadTexture2D("D:/workspace/Ilum/Asset/Texture/Material/streaky-metal/streaky-metal1_albedo.png");
	auto &image2 = Ilum::Resource::ResourceCache::LoadTexture2D("D:/workspace/Ilum/Asset/Texture/Material/streaky-metal/streaky-metal1_ao.png");
	auto &image3 = Ilum::Resource::ResourceCache::LoadTexture2D("D:/workspace/Ilum/Asset/Texture/Material/streaky-metal/streaky-metal1_metallic.png");
	auto &image4 = Ilum::Resource::ResourceCache::LoadTexture2D("D:/workspace/Ilum/Asset/Texture/Material/streaky-metal/streaky-metal1_normal-ogl.png");
	Ilum::Resource::ResourceCache::LoadTexture2DAsync("D:/workspace/Ilum/Asset/Texture/Material/streaky-metal/streaky-metal1_roughness.png");

	Ilum::Resource::ResourceCache::RemoveTexture2D("D:/workspace/Ilum/Asset/Texture/Material/streaky-metal/streaky-metal1_ao.png");
	Ilum::Resource::ResourceCache::RemoveTexture2D("D:/workspace/Ilum/Asset/Texture/Material/streaky-metal/streaky-metal1_normal-ogl.png");

	Ilum::Resource::ResourceCache::OnUpdate();

	while (Ilum::Resource::ResourceCache::IsTexture2DLoading())
	{
		Ilum::Resource::ResourceCache::OnUpdate();
	}

	Ilum::Resource::ResourceCache::ClearAll();

	return 0;
}