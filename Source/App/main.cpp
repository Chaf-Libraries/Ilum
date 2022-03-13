#include <Ilum/Device/Window.hpp>
#include <Ilum/Editor/Editor.hpp>
#include <Ilum/Engine/Context.hpp>
#include <Ilum/Engine/Engine.hpp>
#include <Ilum/Graphics/Shader/Shader.hpp>
#include <Ilum/Renderer/Renderer.hpp>
#include <Ilum/Scene/Scene.hpp>
#include <Ilum/Scene/System.hpp>
#include <Ilum/Timing/Timer.hpp>

#include "Geometry/Mesh/FMesh.hpp"
#include "Geometry/Mesh/HEMesh.hpp"

int main()
{
	Ilum::Engine engine;

	Ilum::Window::instance()->setIcon(std::string(PROJECT_SOURCE_DIR) + "Asset/Texture/Icon/logo.bmp");

	auto model = Ilum::Renderer::instance()->getResourceCache().loadModel(std::string(PROJECT_SOURCE_DIR) + "Asset/Model/head.obj");
	std::vector<glm::vec3> vertices;
	vertices.reserve(model.get().vertices.size());
	for (auto &v : model.get().vertices)
	{
		vertices.push_back(v.position);
	}
	Ilum::geometry::HEMesh hemesh(vertices, model.get().indices);

	uint32_t boundary_count = 0;

	for (auto* v : hemesh.vertices())
	{
		if (hemesh.onBoundary(v))
		{
			boundary_count++;
		}
	}

	auto boundaries = hemesh.boundary();

	while (!Ilum::Window::instance()->shouldClose())
	{
		engine.onTick();

		Ilum::Window::instance()->setTitle((Ilum::Scene::instance()->name.empty() ? "IlumEngine" : Ilum::Scene::instance()->name) + " FPS: " + std::to_string(Ilum::Timer::instance()->getFPS()));
	}

	return 0;
}