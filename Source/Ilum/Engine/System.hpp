#pragma once

#include <Renderer/Renderer.hpp>
#include <Resource/ResourceManager.hpp>
#include <Scene/Component/HierarchyComponent.hpp>
#include <Scene/Component/StaticMeshComponent.hpp>
#include <Scene/Component/TagComponent.hpp>
#include <Scene/Component/TransformComponent.hpp>
#include <Scene/Scene.hpp>

namespace Ilum
{
class System
{
  public:
	template <typename T1, typename... Tn>
	inline static void Execute(Renderer *renderer, T1 &t1, Tn &...tn)
	{
	}

	template <typename T1, typename... Tn>
	inline static void Execute(Renderer *renderer)
	{
		renderer->GetScene()->GroupExecute<T1, Tn...>([&](entt::entity entity, T1 &t1, Tn &...tn) {
			Execute(renderer, t1, tn...);
		});
	}

	inline static void Tick(Renderer *renderer)
	{
		Execute<TransformComponent, HierarchyComponent>(renderer);
		Execute<StaticMeshComponent>(renderer);
	}
};

template <>
inline void System::Execute<StaticMeshComponent>(Renderer *renderer, StaticMeshComponent &static_mesh)
{
	if (static_mesh.update)
	{
		auto meta = renderer->GetResourceManager()->GetModel(static_mesh.uuid);
		if (meta)
		{
			static_mesh.submeshes = meta->submeshes;
		}
		else
		{
			static_mesh.submeshes.clear();
		}
	}
}

template <>
inline void System::Execute<TransformComponent, HierarchyComponent>(Renderer *renderer, TransformComponent&, HierarchyComponent&)
{
}

// template <>
// inline void System::Execute<TransformComponent, HierarchyComponent>(Renderer *renderer)
//{
//	std::function<void(Entity &)> transform_recrusive = [&](Entity &entity) {
//		if (!entity.IsValid())
//		{
//			return;
//		}
//
//		auto &transform = entity.GetComponent<TransformComponent>();
//		auto &hierarchy = entity.GetComponent<HierarchyComponent>();
//		if (hierarchy.parent != entt::null)
//		{
//			transform.local_transform = glm::scale(glm::translate(glm::mat4(1.f), transform.translation) * glm::mat4_cast(glm::qua<float>(glm::radians(transform.rotation))), transform.scale);
//			transform.world_transform = Entity(renderer->GetScene(), hierarchy.parent).GetComponent<TransformComponent>().world_transform * transform.local_transform;
//		}
//
//		auto child = Entity(renderer->GetScene(), hierarchy.first);
//
//		while (child != entt::null)
//		{
//			transform_recrusive(child);
//			child = Entity(renderer->GetScene(), child.GetComponent<HierarchyComponent>().next);
//		}
//	};
//
//	auto *scene = renderer->GetScene();
//
//	std::vector<entt::entity> roots;
//
//	scene->GroupExecute([&](entt::entity entity, HierarchyComponent &hierarchy, TransformComponent &transform) {
//		if (hierarchy.parent == entt::null)
//		{
//			transform.local_transform = glm::scale(glm::translate(glm::mat4(1.f), transform.translation) * glm::mat4_cast(glm::qua<float>(glm::radians(transform.rotation))), transform.scale);
//			transform.world_transform = transform.local_transform;
//			roots.emplace_back(entity);
//		}
//	});
//
//	//tbb::parallel_for_each(roots.begin(), roots.end(), [&group](auto entity) {
//	//	transform_recrusive(entity);
//	//});
// }

}        // namespace Ilum