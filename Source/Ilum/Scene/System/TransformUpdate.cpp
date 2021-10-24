#include "TransformUpdate.hpp"

#include "Scene/Component/Hierarchy.hpp"
#include "Scene/Component/Transform.hpp"
#include "Scene/Entity.hpp"
#include "Scene/Scene.hpp"

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>

namespace Ilum::sym
{
void TransformUpdate::run()
{
	bool need_update = false;
	auto view        = Scene::instance()->getRegistry().view<cmpt::Transform>();
	std::for_each(std::execution::par_unseq, view.begin(), view.end(), [&view, &need_update](auto entity) {
		auto &transform = Entity(entity).getComponent<cmpt::Transform>();
		if (transform.update)
		{
			transform.local_transform = glm::scale(glm::translate(glm::mat4(1.f), transform.translation) * glm::mat4_cast(glm::qua<float>(glm::radians(transform.rotation))), transform.scale);
			transform.update          = false;
			need_update               = true;
		}
	});

	//auto                      group = Scene::instance()->getRegistry().group<cmpt::Transform, cmpt::Hierarchy>();
	//std::vector<entt::entity> roots;
	//if (need_update)
	//{
	//	// Find roots
	//	for (auto &entity : group)
	//	{
	//		auto &[hierarchy, transform] = group.get<cmpt::Hierarchy, cmpt::Transform>(entity);

	//		if (hierarchy.parent == entt::null)
	//		{
	//			transform.world_transform = transform.local_transform;
	//			roots.emplace_back(entity);
	//		}
	//	}

	//	std::for_each(std::execution::par_unseq, roots.begin(), roots.end(), [&group](auto entity) {
	//		while (entity != entt::null)
	//		{
	//			auto &hierarchy = Entity(entity).getComponent<cmpt::Hierarchy>();
	//			auto  next      = Entity(hierarchy.next);
	//			auto  parent    = next.getComponent<cmpt::Hierarchy>().parent;

	//			auto &transform = next.getComponent<cmpt::Transform>();
	//			transform.world_transform = transform.local_transform;
	//			
	//			if (parent != entt::null && Entity(parent).hasComponent<cmpt::Transform>())
	//			{
	//				auto& transform = next.getComponent<cmpt::Transform>();
	//				transform.world_transform = Entity(parent).getComponent<cmpt::Transform>().world_transform * transform.world_transform;
	//			}
	//			
	//			entity = hierarchy.next;
	//		}
	//	});
	//}

	//for (auto entity : view)
	//{
	//	auto &transform = Entity(entity).getComponent<cmpt::Transform>();
	//}
}
}        // namespace Ilum::sym
