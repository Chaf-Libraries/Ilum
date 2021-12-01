#include "TransformUpdate.hpp"

#include "Scene/Component/Hierarchy.hpp"
#include "Scene/Component/Transform.hpp"
#include "Scene/Entity.hpp"
#include "Scene/Scene.hpp"

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>

#include<tbb/tbb.h>

namespace Ilum::sym
{
inline void transform_recrusive(entt::entity entity)
{
	if (entity == entt::null)
	{
		return;
	}

	auto &transform = Entity(entity).getComponent<cmpt::Transform>();
	auto &hierarchy = Entity(entity).getComponent<cmpt::Hierarchy>();
	if (hierarchy.parent != entt::null)
	{
		transform.local_transform = glm::scale(glm::translate(glm::mat4(1.f), transform.translation) * glm::mat4_cast(glm::qua<float>(glm::radians(transform.rotation))), transform.scale);
		transform.world_transform = Entity(hierarchy.parent).getComponent<cmpt::Transform>().world_transform * transform.local_transform;
	}

	auto child = hierarchy.first;

	while (child != entt::null)
	{
		transform_recrusive(child);
		child = Entity(child).getComponent<cmpt::Hierarchy>().next;
	}
}

void TransformUpdate::run()
{
	if (cmpt::Transform::update)
	{
		cmpt::Transform::update = false;
		auto group              = Scene::instance()->getRegistry().group<>(entt::get<cmpt::Transform, cmpt::Hierarchy>);

		std::vector<entt::entity> roots;

		// Find roots
		for (auto &entity : group)
		{
			auto &[hierarchy, transform] = group.get<cmpt::Hierarchy, cmpt::Transform>(entity);

			if (hierarchy.parent == entt::null)
			{
				transform.local_transform = glm::scale(glm::translate(glm::mat4(1.f), transform.translation) * glm::mat4_cast(glm::qua<float>(glm::radians(transform.rotation))), transform.scale);
				transform.world_transform = transform.local_transform;
				roots.emplace_back(entity);
			}
		}

		tbb::parallel_for_each(roots.begin(), roots.end(), [&group](auto entity) {
			transform_recrusive(entity);
		});
	}
}
}        // namespace Ilum::sym
