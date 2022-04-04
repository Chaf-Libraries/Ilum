#include "SceneSerializer.hpp"
#include "Entity.hpp"
#include "Scene.hpp"

#include "Component/Camera.hpp"
#include "Component/Hierarchy.hpp"
#include "Component/Light.hpp"
#include "Component/Renderable.hpp"
#include "Component/Tag.hpp"
#include "Component/Transform.hpp"

#include "Renderer/Renderer.hpp"

#include "File/FileSystem.hpp"

#include <cereal/archives/binary.hpp>
#include <cereal/archives/xml.hpp>

#include <fstream>

namespace Ilum
{
void SceneSerializer::serialize(const std::string &file_path)
{
	std::ofstream            os(file_path, std::ios::binary);
	cereal::XMLOutputArchive archive(os);

	entt::snapshot{
	    Scene::instance()->getRegistry()}
	    .entities(archive)
	    .component<
	        cmpt::Tag,
	        cmpt::Transform,
	        cmpt::Hierarchy,
	        cmpt::DirectionalLight,
	        cmpt::PointLight,
	        cmpt::SpotLight,
	        cmpt::StaticMeshRenderer,
	        cmpt::DynamicMeshRenderer,
	        cmpt::CurveRenderer,
	        cmpt::SurfaceRenderer,
	        cmpt::PerspectiveCamera,
	        cmpt::OrthographicCamera>(archive);
}

void SceneSerializer::deserialize(const std::string &file_path)
{
	std::ifstream           os(file_path, std::ios::binary);
	cereal::XMLInputArchive archive(os);
	Scene::instance()->clear();
	entt::snapshot_loader{
	    Scene::instance()->getRegistry()}
	    .entities(archive)
	    .component<
	        cmpt::Tag,
	        cmpt::Transform,
	        cmpt::Hierarchy,
	        cmpt::DirectionalLight,
	        cmpt::PointLight,
	        cmpt::SpotLight,
	        cmpt::StaticMeshRenderer,
	        cmpt::DynamicMeshRenderer,
	        cmpt::CurveRenderer,
	        cmpt::SurfaceRenderer,
	        cmpt::PerspectiveCamera,
	        cmpt::OrthographicCamera>(archive);

	const auto static_mesh  = Scene::instance()->getRegistry().group<cmpt::StaticMeshRenderer>(entt::get<cmpt::Transform, cmpt::Tag>);
	const auto dynamic_mesh = Scene::instance()->getRegistry().group<cmpt::DynamicMeshRenderer>(entt::get<cmpt::Transform, cmpt::Tag>);

	static_mesh.each([](const entt::entity &entity, const cmpt::StaticMeshRenderer &mesh, const cmpt::Transform &transform, const cmpt::Tag &tag) {
		Renderer::instance()->getResourceCache().loadModelAsync(mesh.model);
		for (const auto &mat : mesh.materials)
		{
			for (const auto &tex : mat.textures)
			{
				Renderer::instance()->getResourceCache().loadImageAsync(tex);
			}
		}
	});

	dynamic_mesh.each([](const entt::entity &entity, const cmpt::DynamicMeshRenderer &mesh, const cmpt::Transform &transform, const cmpt::Tag &tag) {
		for (const auto &tex : mesh.material.textures)
		{
			Renderer::instance()->getResourceCache().loadImageAsync(tex);
		}
	});

	cmpt::Transform::update = true;
}
}        // namespace Ilum