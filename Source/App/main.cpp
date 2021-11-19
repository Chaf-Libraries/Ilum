#include <Ilum/Device/PhysicalDevice.hpp>
#include <Ilum/Device/Window.hpp>
#include <Ilum/Editor/Editor.hpp>
#include <Ilum/Engine/Context.hpp>
#include <Ilum/Engine/Engine.hpp>
#include <Ilum/Graphics/GraphicsContext.hpp>
#include <Ilum/Renderer/RenderPass/BloomPass.hpp>
#include <Ilum/Renderer/RenderPass/GeometryPass.hpp>
#include <Ilum/Renderer/RenderPass/LightPass.hpp>
#include <Ilum/Renderer/RenderPass/TonemappingPass.hpp>
#include <Ilum/Renderer/Renderer.hpp>
#include <Ilum/Scene/Component/Hierarchy.hpp>
#include <Ilum/Scene/Component/MeshRenderer.hpp>
#include <Ilum/Scene/Component/Tag.hpp>
#include <Ilum/Scene/Component/Transform.hpp>
#include <Ilum/Scene/Scene.hpp>
#include <Ilum/Scene/System.hpp>
#include <Ilum/Threading/ThreadPool.hpp>
#include <Ilum/Timing/Timer.hpp>

int main()
{
	Ilum::Engine engine;

	Ilum::Renderer::instance()->buildRenderGraph = [](Ilum::RenderGraphBuilder &builder) {
		builder
		    .addRenderPass("GeometryPass", std::make_unique<Ilum::pass::GeometryPass>())
		    .addRenderPass("LightPass", std::make_unique<Ilum::pass::LightPass>())
		    .addRenderPass("BloomPass", std::make_unique<Ilum::pass::BloomPass>())
		    .addRenderPass("Tonemapping", std::make_unique<Ilum::pass::TonemappingPass>("blooming"))

		    .setView("gbuffer - normal")
		    .setOutput("gbuffer - normal");
	};

	Ilum::Renderer::instance()->rebuild();

	Ilum::Window::instance()->setIcon(std::string(PROJECT_SOURCE_DIR) + "Asset/Texture/Icon/logo.bmp");

	while (!Ilum::Window::instance()->shouldClose())
	{
		engine.onTick();

		Ilum::Window::instance()->setTitle((Ilum::Scene::instance()->name.empty() ? "IlumEngine" : Ilum::Scene::instance()->name) + " FPS: " + std::to_string(Ilum::Timer::instance()->getFPS()));
	}

	return 0;
}