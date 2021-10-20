#include "TransformUpdate.hpp"

#include "Scene/Component/Transform.hpp"
#include "Scene/Scene.hpp"
#include "Scene/Entity.hpp"

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>

namespace Ilum::sym
{
void TransformUpdate::run()
{
	auto view = Scene::instance()->getRegistry().view<cmpt::Transform>();
	std::for_each(std::execution::par_unseq, view.begin(), view.end(), [&view](auto entity) {
		auto &transform           = Entity(entity).getComponent<cmpt::Transform>();
		transform.world_transform = glm::scale(glm::translate(glm::mat4(1.f), transform.position)*glm::mat4_cast(glm::qua<float>(glm::radians(transform.rotation))), transform.scale);
	});

	//for (auto entity : view)
	//{
	//	auto &transform = Entity(entity).getComponent<cmpt::Transform>();
	//}
}
}        // namespace Ilum::sym
