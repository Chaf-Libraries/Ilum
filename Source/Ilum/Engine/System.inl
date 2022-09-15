#pragma once

#include "System.hpp"

namespace Ilum
{
template <>
inline void System::Execute<TransformComponent, HierarchyComponent>(Renderer *renderer)
{
	std::function<void(Entity &)> transform_recrusive = [&](Entity &entity) {
		if (!entity.IsValid())
		{
			return;
		}

		auto  &transform = entity.GetComponent<TransformComponent>();
		auto  &hierarchy = entity.GetComponent<HierarchyComponent>();
		Entity parent(renderer->GetScene(), hierarchy.parent);

		if (parent.IsValid())
		{
			auto &parent_transform = parent.GetComponent<TransformComponent>();
			auto &parent_hierarchy = parent.GetComponent<HierarchyComponent>();
			if (parent_transform.update || parent_hierarchy.update || transform.update || hierarchy.update)
			{
				transform.local_transform = glm::scale(glm::translate(glm::mat4(1.f), transform.translation) * glm::mat4_cast(glm::qua<float>(glm::radians(transform.rotation))), transform.scale);
				transform.world_transform = parent.GetComponent<TransformComponent>().world_transform * transform.local_transform;
				renderer->UpdateGPUScene();
			}
		}
		else
		{
			if (transform.update || hierarchy.update)
			{
				transform.local_transform = glm::scale(glm::translate(glm::mat4(1.f), transform.translation) * glm::mat4_cast(glm::qua<float>(glm::radians(transform.rotation))), transform.scale);
				transform.world_transform = transform.local_transform;
				renderer->UpdateGPUScene();
			}
		}

		auto child = Entity(renderer->GetScene(), hierarchy.first);

		while (child.IsValid())
		{
			transform_recrusive(child);
			child = Entity(renderer->GetScene(), child.GetComponent<HierarchyComponent>().next);
		}
	};

	auto *scene = renderer->GetScene();

	std::vector<entt::entity> roots;

	scene->GroupExecute<HierarchyComponent, TransformComponent>([&](entt::entity entity, HierarchyComponent &hierarchy, TransformComponent &transform) {
		if (hierarchy.parent == entt::null)
		{
			transform.local_transform = glm::scale(glm::translate(glm::mat4(1.f), transform.translation) * glm::mat4_cast(glm::qua<float>(glm::radians(transform.rotation))), transform.scale);
			transform.world_transform = transform.local_transform;
			roots.emplace_back(entity);
		}
	});

	for (auto &root : roots)
	{
		Entity entity(renderer->GetScene(), root);
		transform_recrusive(entity);
	}
}

template <>
inline void System::Execute<StaticMeshComponent>(Renderer *renderer)
{
	renderer->GetScene()->GroupExecute<StaticMeshComponent>([&](entt::entity entity, StaticMeshComponent &static_mesh) {
		if (static_mesh.update)
		{
			auto meta = renderer->GetResourceManager()->GetModel(static_mesh.uuid);
			if (meta)
			{
				static_mesh.materials.resize(meta->submeshes.size());
				std::fill(static_mesh.materials.begin(), static_mesh.materials.end(), "");
			}
			else
			{
				static_mesh.materials.clear();
			}
			renderer->UpdateGPUScene();
		}
	});
}
}