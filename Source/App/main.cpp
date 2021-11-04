#include <Ilum/Device/PhysicalDevice.hpp>
#include <Ilum/Device/Window.hpp>
#include <Ilum/Editor/Editor.hpp>
#include <Ilum/Engine/Context.hpp>
#include <Ilum/Engine/Engine.hpp>
#include <Ilum/Graphics/GraphicsContext.hpp>
#include <Ilum/Graphics/Pipeline/Shader.hpp>
#include <Ilum/Graphics/Pipeline/ShaderCache.hpp>
#include <Ilum/Loader/ModelLoader/ModelLoader.hpp>
#include <Ilum/Material/BlinnPhong.h>
#include <Ilum/Renderer/RenderPass/DefaultPass.hpp>
#include <Ilum/Renderer/RenderPass/GeometryPass.hpp>
#include <Ilum/Renderer/Renderer.hpp>
#include <Ilum/Scene/Component/Camera.hpp>
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

		    .addRenderPass("DefaultPass", std::make_unique<Ilum::pass::DefaultPass>("result"))
		    .setView("result")
		    .setOutput("result");

		//.setView("gbuffer - normal")
		//    .setOutput("gbuffer - normal");
	};


	std::unordered_map<std::string, uint32_t> test_map;
	auto                                      t = test_map["a"];
	LOG_INFO("t = {}", t);

	test_map.emplace("a", 10);

	for (auto& [key, val] : test_map)
	{
		LOG_INFO("Key: {}, Val: {}", key, val);
	}




	Ilum::Renderer::instance()->rebuild();

	auto title = Ilum::Window::instance()->getTitle();
	while (!Ilum::Window::instance()->shouldClose())
	{
		engine.onTick();

		Ilum::Window::instance()->setTitle(title + " FPS: " + std::to_string(Ilum::Timer::instance()->getFPS()));
	}

	return 0;
}