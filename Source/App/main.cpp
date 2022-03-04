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
#include "Geometry/Mesh/EMesh.hpp"

int main()
{
	Ilum::Engine engine;

	Ilum::Window::instance()->setIcon(std::string(PROJECT_SOURCE_DIR) + "Asset/Texture/Icon/logo.bmp");

	auto model = Ilum::Renderer::instance()->getResourceCache().loadModel(std::string(PROJECT_SOURCE_DIR) + "Asset/Model/grid.obj");
	std::vector<glm::vec3> vertices;
	vertices.reserve(model.get().vertices.size());
	for (auto &v : model.get().vertices)
	{
		vertices.push_back(v.position);
	}
	Ilum::geometry::EMesh mesh(vertices, model.get().indices);

	const auto &mesh_vertices = mesh.vertices();
	const auto &mesh_faces = mesh.faces();



	while (!Ilum::Window::instance()->shouldClose())
	{
		engine.onTick();

		Ilum::Window::instance()->setTitle((Ilum::Scene::instance()->name.empty() ? "IlumEngine" : Ilum::Scene::instance()->name) + " FPS: " + std::to_string(Ilum::Timer::instance()->getFPS()));
	}

	return 0;
}