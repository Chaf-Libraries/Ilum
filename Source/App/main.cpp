#include <Ilum/Device/Window.hpp>
#include <Ilum/Editor/Editor.hpp>
#include <Ilum/Engine/Context.hpp>
#include <Ilum/Engine/Engine.hpp>
#include <Ilum/Graphics/GraphicsContext.hpp>
#include <Ilum/Graphics/Pipeline/Shader.hpp>
#include <Ilum/Graphics/Pipeline/ShaderCache.hpp>
#include <Ilum/Loader/ModelLoader/ModelLoader.hpp>
#include <Ilum/Renderer/Renderer.hpp>
#include <Ilum/Scene/Component/Camera.hpp>
#include <Ilum/Scene/Component/Hierarchy.hpp>
#include <Ilum/Scene/Component/Tag.hpp>
#include <Ilum/Scene/Component/Transform.hpp>
#include <Ilum/Scene/Component/MeshRenderer.hpp>
#include <Ilum/Scene/Scene.hpp>
#include <Ilum/Scene/System.hpp>
#include <Ilum/Timing/Timer.hpp>
#include <Ilum/Renderer/Renderer.hpp>
#include <Ilum/Renderer/RenderPass/GeometryPass.hpp>

int main()
{
	Ilum::Engine engine;

	for (auto i = 0; i < 10; i++)
	{
		auto entity = Ilum::Scene::instance()->createEntity("test" + std::to_string(i));
	}

	auto entity = Ilum::Scene::instance()->createEntity("test" + std::to_string(10));
	entity.addComponent<Ilum::cmpt::MeshRenderer>().model = "../Asset/Model/head.obj";
	auto view   = Ilum::Scene::instance()->getRegistry().view<Ilum::cmpt::Tag>();

	auto model = Ilum::Renderer::instance()->getResourceCache().loadModel("../Asset/Model/head.obj");

	Ilum::Renderer::instance()->getResourceCache().loadImage("../Asset/Texture/bricks2.jpg");
	Ilum::Renderer::instance()->getResourceCache().loadImage("../Asset/Texture/bricks2_disp.jpg");
	Ilum::Renderer::instance()->getResourceCache().loadImage("../Asset/Texture/bricks2_normal.jpg");
	Ilum::Renderer::instance()->getResourceCache().loadImage("../Asset/Texture/brickwall.jpg");
	Ilum::Renderer::instance()->getResourceCache().loadImage("../Asset/Texture/brickwall_normal.jpg");
	Ilum::Renderer::instance()->getResourceCache().loadImage("../Asset/Texture/cg_displacementmap.jpg");
	Ilum::Renderer::instance()->getResourceCache().loadImage("../Asset/Texture/cg_displacementmap__.jpg");
	Ilum::Renderer::instance()->getResourceCache().loadImage("../Asset/Texture/cg_normalmap.jpg");
	Ilum::Renderer::instance()->getResourceCache().loadImage("../Asset/Texture/cg_normalmap__.jpg");
	Ilum::Renderer::instance()->getResourceCache().loadImage("../Asset/Texture/face.png");

	//Ilum::Renderer::instance()->setDebug(false);
	//Ilum::Renderer::instance()->setImGui(false);

	Ilum::Renderer::instance()->buildRenderGraph = [](Ilum::RenderGraphBuilder &builder) {
		builder.addRenderPass("GeometryPass", std::make_unique<Ilum::pass::GeometryPass>("gbuffer - normal")).setView("gbuffer - normal").setOutput("gbuffer - normal");
	};

	Ilum::Renderer::instance()->rebuild();
	//for (auto& iter : view)
	//{
	//	LOG_INFO(iter.getComponent<Ilum::cmpt::Tag>().name);
	//}

	//std::for_each(std::execution::par_unseq, view.begin(), view.end(), [&view](auto entity) {
	//	std::cout << std::this_thread::get_id() << std::endl;
	//});

	//auto t = entity.hasComponent<Ilum::cmpt::Tag>();
	auto title = Ilum::Window::instance()->getTitle();
	while (!Ilum::Window::instance()->shouldClose())
	{
		engine.onTick();

		Ilum::Window::instance()->setTitle(title + " FPS: " + std::to_string(Ilum::Timer::instance()->getFPS()));
	}

	return 0;
}