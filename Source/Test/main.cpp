#include <iostream>
#include <SceneGraph/Scene.hpp>
#include <SceneGraph/Node.hpp>
#include <Components/Transform.hpp>

using namespace Ilum;

int main()
{
	Scene scene;

	auto* node = scene.CreateNode("Test");
	node->AddComponent(std::make_unique<Cmpt::Transform>(node));
	scene.EraseNode(node);

	std::cout << "Fuck";
	return 0;
}