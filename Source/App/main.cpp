#include <Ilum/Engine/Context.hpp>
#include <Ilum/Engine/Engine.hpp>
#include <Ilum/Scene/Scene.hpp>
#include <Ilum/Scene/System.hpp>

#include <Graphics/Device/Window.hpp>
#include <Graphics/RenderContext.hpp>

#include <Resource/ResourceCache.hpp>

int main()
{
	Ilum::Engine engine;

	Ilum::Graphics::RenderContext::GetWindow().SetIcon(std::string(PROJECT_SOURCE_DIR) + "Asset/Texture/Icon/logo.bmp");

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

	while (!Ilum::Graphics::RenderContext::GetWindow().ShouldClose())
	{
		engine.onTick();

		Ilum::Graphics::RenderContext::GetWindow().SetTitle((Ilum::Scene::instance()->name.empty() ? "IlumEngine" : Ilum::Scene::instance()->name));
	}

	return 0;
}