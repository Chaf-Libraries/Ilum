#include "SceneSerializer.hpp"
#include "Entity.hpp"
#include "Scene.hpp"

#include "Component/DirectionalLight.hpp"
#include "Component/Hierarchy.hpp"
#include "Component/Light.hpp"
#include "Component/MeshRenderer.hpp"
#include "Component/PointLight.hpp"
#include "Component/SpotLight.hpp"
#include "Component/Tag.hpp"
#include "Component/Transform.hpp"

#include "Renderer/Renderer.hpp"

#include "File/FileSystem.hpp"

#include "Material/DisneyPBR.h"
#include "Material/Material.h"

#include <fstream>

namespace Ilum
{
static std::unordered_set<std::string> load_images;
static std::unordered_set<std::string> load_models;

template <typename T>
void serialize_component(YAML::Emitter &emitter, const entt::entity entity)
{
}

template <typename T>
void serialize_material(YAML::Emitter &emitter, const T &material)
{
}

template <>
void serialize_material<material::DisneyPBR>(YAML::Emitter &emitter, const material::DisneyPBR &material)
{
	emitter << YAML::BeginMap;
	emitter << YAML::Key << "type" << YAML::Value << typeid(material::DisneyPBR).name();
	emitter << YAML::Key << "base_color" << YAML::Value << material.base_color;
	emitter << YAML::Key << "emissive_color" << YAML::Value << material.emissive_color;
	emitter << YAML::Key << "emissive_intensity" << YAML::Value << material.emissive_intensity;
	emitter << YAML::Key << "metallic_factor" << YAML::Value << material.metallic_factor;
	emitter << YAML::Key << "roughness_factor" << YAML::Value << material.roughness_factor;
	emitter << YAML::Key << "displacement_height" << YAML::Value << material.displacement_height;
	emitter << YAML::Key << "albedo_map" << YAML::Value << FileSystem::getRelativePath(material.albedo_map);
	emitter << YAML::Key << "normal_map" << YAML::Value << FileSystem::getRelativePath(material.normal_map);
	emitter << YAML::Key << "metallic_map" << YAML::Value << FileSystem::getRelativePath(material.metallic_map);
	emitter << YAML::Key << "roughness_map" << YAML::Value << FileSystem::getRelativePath(material.roughness_map);
	emitter << YAML::Key << "emissive_map" << YAML::Value << FileSystem::getRelativePath(material.emissive_map);
	emitter << YAML::Key << "ao_map" << YAML::Value << FileSystem::getRelativePath(material.ao_map);
	emitter << YAML::Key << "displacement_map" << YAML::Value << FileSystem::getRelativePath(material.displacement_map);
	emitter << YAML::EndMap;
}

void serialize_material(YAML::Emitter &emitter, scope<IMaterial> &material)
{
	if (material->type() == typeid(material::DisneyPBR))
	{
		serialize_material<material::DisneyPBR>(emitter, *static_cast<material::DisneyPBR *>(material.get()));
	}
}

template <>
void serialize_component<cmpt::Tag>(YAML::Emitter &emitter, const entt::entity entity)
{
	if (!Entity(entity).hasComponent<cmpt::Tag>())
	{
		return;
	}

	auto &tag = Entity(entity).getComponent<cmpt::Tag>();

	emitter << YAML::Key << typeid(cmpt::Tag).name();
	emitter << YAML::BeginMap;
	emitter << YAML::Key << "name" << YAML::Value << tag.name;
	emitter << YAML::Key << "active" << YAML::Value << tag.active;
	emitter << YAML::EndMap;
}

template <>
void serialize_component<cmpt::Transform>(YAML::Emitter &emitter, const entt::entity entity)
{
	if (!Entity(entity).hasComponent<cmpt::Transform>())
	{
		return;
	}

	auto &transform = Entity(entity).getComponent<cmpt::Transform>();

	emitter << YAML::Key << typeid(cmpt::Transform).name();
	emitter << YAML::BeginMap;
	emitter << YAML::Key << "translation" << YAML::Value << transform.translation;
	emitter << YAML::Key << "rotation" << YAML::Value << transform.rotation;
	emitter << YAML::Key << "scale" << YAML::Value << transform.scale;
	emitter << YAML::Key << "local_transform" << YAML::Value << transform.local_transform;
	emitter << YAML::Key << "world_transform" << YAML::Value << transform.local_transform;
	emitter << YAML::EndMap;
}

template <>
void serialize_component<cmpt::Hierarchy>(YAML::Emitter &emitter, const entt::entity entity)
{
	if (!Entity(entity).hasComponent<cmpt::Hierarchy>())
	{
		return;
	}

	auto &hierarchy = Entity(entity).getComponent<cmpt::Hierarchy>();

	emitter << YAML::Key << typeid(cmpt::Hierarchy).name();
	emitter << YAML::BeginMap;
	emitter << YAML::Key << "parent" << YAML::Value << static_cast<uint32_t>(hierarchy.parent);
	emitter << YAML::Key << "first" << YAML::Value << static_cast<uint32_t>(hierarchy.first);
	emitter << YAML::Key << "next" << YAML::Value << static_cast<uint32_t>(hierarchy.next);
	emitter << YAML::Key << "prev" << YAML::Value << static_cast<uint32_t>(hierarchy.prev);
	emitter << YAML::EndMap;
}

template <>
void serialize_component<cmpt::MeshRenderer>(YAML::Emitter &emitter, const entt::entity entity)
{
	if (!Entity(entity).hasComponent<cmpt::MeshRenderer>())
	{
		return;
	}

	auto &mesh_renderer = Entity(entity).getComponent<cmpt::MeshRenderer>();

	emitter << YAML::Key << typeid(cmpt::MeshRenderer).name();
	emitter << YAML::BeginMap;
	emitter << YAML::Key << "model" << YAML::Value << FileSystem::getRelativePath(mesh_renderer.model);
	emitter << YAML::Key << "materials" << YAML::Value;
	emitter << YAML::BeginSeq;
	for (auto &material : mesh_renderer.materials)
	{
		serialize_material(emitter, material);
	}
	emitter << YAML::EndSeq;
	emitter << YAML::EndMap;
}

template <>
void serialize_component<cmpt::DirectionalLight>(YAML::Emitter &emitter, const entt::entity entity)
{
	auto *light = static_cast<cmpt::DirectionalLight *>(Entity(entity).getComponent<cmpt::Light>().impl.get());

	emitter << YAML::Key << "type" << YAML::Value << static_cast<uint32_t>(light->type());
	emitter << YAML::Key << "color" << YAML::Value << light->data.color;
	emitter << YAML::Key << "intensity" << YAML::Value << light->data.intensity;
	emitter << YAML::Key << "direction" << YAML::Value << light->data.direction;
}

template <>
void serialize_component<cmpt::PointLight>(YAML::Emitter &emitter, const entt::entity entity)
{
	auto *light = static_cast<cmpt::PointLight *>(Entity(entity).getComponent<cmpt::Light>().impl.get());

	emitter << YAML::Key << "type" << YAML::Value << static_cast<uint32_t>(light->type());
	emitter << YAML::Key << "color" << YAML::Value << light->data.color;
	emitter << YAML::Key << "intensity" << YAML::Value << light->data.intensity;
	emitter << YAML::Key << "position" << YAML::Value << light->data.position;
	emitter << YAML::Key << "constant" << YAML::Value << light->data.constant;
	emitter << YAML::Key << "linear" << YAML::Value << light->data.linear;
	emitter << YAML::Key << "quadratic" << YAML::Value << light->data.quadratic;
}

template <>
void serialize_component<cmpt::SpotLight>(YAML::Emitter &emitter, const entt::entity entity)
{
	auto *light = static_cast<cmpt::SpotLight *>(Entity(entity).getComponent<cmpt::Light>().impl.get());

	emitter << YAML::Key << "type" << YAML::Value << static_cast<uint32_t>(light->type());
	emitter << YAML::Key << "color" << YAML::Value << light->data.color;
	emitter << YAML::Key << "intensity" << YAML::Value << light->data.intensity;
	emitter << YAML::Key << "position" << YAML::Value << light->data.position;
	emitter << YAML::Key << "cut_off" << YAML::Value << light->data.cut_off;
	emitter << YAML::Key << "direction" << YAML::Value << light->data.direction;
	emitter << YAML::Key << "outer_cut_off" << YAML::Value << light->data.outer_cut_off;
}

template <>
void serialize_component<cmpt::Light>(YAML::Emitter &emitter, const entt::entity entity)
{
	if (!Entity(entity).hasComponent<cmpt::Light>())
	{
		return;
	}

	auto &light = Entity(entity).getComponent<cmpt::Light>();

	emitter << YAML::Key << typeid(cmpt::Light).name();
	emitter << YAML::BeginMap;
	switch (light.type)
	{
		case cmpt::LightType::Directional:
			serialize_component<cmpt::DirectionalLight>(emitter, entity);
			break;
		case cmpt::LightType::Point:
			serialize_component<cmpt::PointLight>(emitter, entity);
			break;
		case cmpt::LightType::Spot:
			serialize_component<cmpt::SpotLight>(emitter, entity);
			break;
		default:
			break;
	}
	emitter << YAML::EndMap;
}

void serialize_entity(YAML::Emitter &emitter, const entt::entity entity)
{
	if (entity == entt::null)
	{
		return;
	}

	emitter << YAML::BeginMap;
	emitter << YAML::Key << "UUID" << YAML::Value << static_cast<uint32_t>(entity);
	serialize_component<cmpt::Tag>(emitter, entity);
	serialize_component<cmpt::Transform>(emitter, entity);
	serialize_component<cmpt::Hierarchy>(emitter, entity);
	serialize_component<cmpt::MeshRenderer>(emitter, entity);
	serialize_component<cmpt::Light>(emitter, entity);
	emitter << YAML::EndMap;
}

void serialize_main_camera(YAML::Emitter &emitter)
{
	emitter << YAML::BeginMap;
	const auto &main_camera = Renderer::instance()->Main_Camera;
	emitter << YAML::Key << "type" << YAML::Value << static_cast<uint32_t>(main_camera.type);
	emitter << YAML::Key << "aspect" << YAML::Value << main_camera.aspect;
	emitter << YAML::Key << "fov" << YAML::Value << main_camera.fov;
	emitter << YAML::Key << "far_plane" << YAML::Value << main_camera.far_plane;
	emitter << YAML::Key << "near_plane" << YAML::Value << main_camera.near_plane;
	emitter << YAML::Key << "forward" << YAML::Value << main_camera.forward;
	emitter << YAML::Key << "position" << YAML::Value << main_camera.position;
	emitter << YAML::Key << "view" << YAML::Value << main_camera.view;
	emitter << YAML::Key << "projection" << YAML::Value << main_camera.projection;
	emitter << YAML::Key << "view_projection" << YAML::Value << main_camera.view_projection;
	emitter << YAML::EndMap;
}

void serialize_color_correction(YAML::Emitter &emitter)
{
	emitter << YAML::BeginMap;
	emitter << YAML::Key << "exposure" << YAML::Value << Renderer::instance()->Color_Correction.exposure;
	emitter << YAML::Key << "gamma" << YAML::Value << Renderer::instance()->Color_Correction.gamma;
	emitter << YAML::EndMap;
}

void serialize_bloom(YAML::Emitter &emitter)
{
	emitter << YAML::BeginMap;
	emitter << YAML::Key << "threshold" << YAML::Value << Renderer::instance()->Bloom.threshold;
	emitter << YAML::Key << "scale" << YAML::Value << Renderer::instance()->Bloom.scale;
	emitter << YAML::Key << "strength" << YAML::Value << Renderer::instance()->Bloom.strength;
	emitter << YAML::Key << "enable" << YAML::Value << Renderer::instance()->Bloom.enable;
	emitter << YAML::EndMap;
}

void deserialize_main_camera(const YAML::Node &data)
{
	Renderer::instance()->Main_Camera.type            = static_cast<Camera::Type>(data["type"].as<uint32_t>());
	Renderer::instance()->Main_Camera.fov             = data["fov"].as<float>();
	Renderer::instance()->Main_Camera.far_plane       = data["far_plane"].as<float>();
	Renderer::instance()->Main_Camera.near_plane      = data["near_plane"].as<float>();
	Renderer::instance()->Main_Camera.forward         = data["forward"].as<glm::vec3>();
	Renderer::instance()->Main_Camera.position        = data["position"].as<glm::vec3>();
	Renderer::instance()->Main_Camera.update          = true;
}

void deserialize_color_correction(const YAML::Node &data)
{
	Renderer::instance()->Color_Correction.exposure = data["exposure"].as<float>();
	Renderer::instance()->Color_Correction.gamma    = data["gamma"].as<float>();
}

void deserialize_bloom(const YAML::Node &data)
{
	Renderer::instance()->Bloom.threshold = data["threshold"].as<float>();
	Renderer::instance()->Bloom.scale     = data["scale"].as<float>();
	Renderer::instance()->Bloom.strength  = data["strength"].as<float>();
	Renderer::instance()->Bloom.enable    = data["enable"].as<uint32_t>();
}

template <typename T>
void deserialize_component(Entity entity, const YAML::Node &data)
{
}

template <>
void deserialize_component<cmpt::Tag>(Entity entity, const YAML::Node &data)
{
	auto &tag  = entity.getComponent<cmpt::Tag>();
	tag.name   = data["name"].as<std::string>();
	tag.active = data["active"].as<bool>();
}

template <>
void deserialize_component<cmpt::Transform>(Entity entity, const YAML::Node &data)
{
	auto &transform           = entity.getComponent<cmpt::Transform>();
	transform.translation     = data["translation"].as<glm::vec3>();
	transform.rotation        = data["rotation"].as<glm::vec3>();
	transform.scale           = data["scale"].as<glm::vec3>();
	transform.local_transform = data["local_transform"].as<glm::mat4>();
	transform.world_transform = data["world_transform"].as<glm::mat4>();
	transform.update          = true;
}

template <>
void deserialize_component<cmpt::Hierarchy>(Entity entity, const YAML::Node &data)
{
	auto &hierarchy  = entity.getComponent<cmpt::Hierarchy>();
	hierarchy.first  = static_cast<entt::entity>(data["first"].as<uint32_t>());
	hierarchy.next   = static_cast<entt::entity>(data["next"].as<uint32_t>());
	hierarchy.parent = static_cast<entt::entity>(data["parent"].as<uint32_t>());
	hierarchy.prev   = static_cast<entt::entity>(data["prev"].as<uint32_t>());
}

template <typename T>
void deserialize_material(Entity entity, const YAML::Node &data)
{
}

template <>
void deserialize_material<material::DisneyPBR>(Entity entity, const YAML::Node &data)
{
	auto *material                = static_cast<material::DisneyPBR *>(entity.getComponent<cmpt::MeshRenderer>().materials.emplace_back(createScope<material::DisneyPBR>()).get());
	material->base_color          = data["base_color"].as<glm::vec4>();
	material->emissive_color      = data["emissive_color"].as<glm::vec3>();
	material->emissive_intensity  = data["emissive_intensity"].as<float>();
	material->metallic_factor     = data["metallic_factor"].as<float>();
	material->roughness_factor    = data["roughness_factor"].as<float>();
	material->displacement_height = data["displacement_height"].as<float>();
	material->albedo_map          = data["albedo_map"].as<std::string>();
	material->normal_map          = data["normal_map"].as<std::string>();
	material->metallic_map        = data["metallic_map"].as<std::string>();
	material->roughness_map       = data["roughness_map"].as<std::string>();
	material->emissive_map        = data["emissive_map"].as<std::string>();
	material->ao_map              = data["ao_map"].as<std::string>();
	material->displacement_map    = data["displacement_map"].as<std::string>();

	Renderer::instance()->getResourceCache().loadImageAsync(material->albedo_map);
	Renderer::instance()->getResourceCache().loadImageAsync(material->normal_map);
	Renderer::instance()->getResourceCache().loadImageAsync(material->metallic_map);
	Renderer::instance()->getResourceCache().loadImageAsync(material->roughness_map);
	Renderer::instance()->getResourceCache().loadImageAsync(material->emissive_map);
	Renderer::instance()->getResourceCache().loadImageAsync(material->ao_map);
	Renderer::instance()->getResourceCache().loadImageAsync(material->displacement_map);
}

void deserialize_material(Entity entity, const YAML::Node &data)
{
	if (data["type"].as<std::string>() == typeid(material::DisneyPBR).name())
	{
		deserialize_material<material::DisneyPBR>(entity, data);
	}
}

template <>
void deserialize_component<cmpt::MeshRenderer>(Entity entity, const YAML::Node &data)
{
	if (!data)
	{
		return;
	}

	auto &mesh_renderer = entity.addComponent<cmpt::MeshRenderer>();
	mesh_renderer.model = data["model"].as<std::string>();
	Renderer::instance()->getResourceCache().loadModelAsync(mesh_renderer.model);
	for (auto material : data["materials"])
	{
		deserialize_material(entity, material);
	}
}

template <>
void deserialize_component<cmpt::DirectionalLight>(Entity entity, const YAML::Node &data)
{
	auto &light                       = entity.getComponent<cmpt::Light>();
	light.impl                        = createScope<cmpt::DirectionalLight>();
	auto *directional_light           = static_cast<cmpt::DirectionalLight *>(light.impl.get());
	directional_light->data.color     = data["color"].as<glm::vec3>();
	directional_light->data.intensity = data["intensity"].as<float>();
	directional_light->data.direction = data["direction"].as<glm::vec3>();
}

template <>
void deserialize_component<cmpt::PointLight>(Entity entity, const YAML::Node &data)
{
	auto &light                 = entity.getComponent<cmpt::Light>();
	light.impl                  = createScope<cmpt::PointLight>();
	auto *point_light           = static_cast<cmpt::PointLight *>(light.impl.get());
	point_light->data.color     = data["color"].as<glm::vec3>();
	point_light->data.intensity = data["intensity"].as<float>();
	point_light->data.position  = data["position"].as<glm::vec3>();
	point_light->data.constant  = data["constant"].as<float>();
	point_light->data.linear    = data["linear"].as<float>();
	point_light->data.quadratic = data["quadratic"].as<float>();
}

template <>
void deserialize_component<cmpt::SpotLight>(Entity entity, const YAML::Node &data)
{
	auto &light                    = entity.getComponent<cmpt::Light>();
	light.impl                     = createScope<cmpt::SpotLight>();
	auto *spot_light               = static_cast<cmpt::SpotLight *>(light.impl.get());
	spot_light->data.color         = data["color"].as<glm::vec3>();
	spot_light->data.intensity     = data["intensity"].as<float>();
	spot_light->data.direction     = data["direction"].as<glm::vec3>();
	spot_light->data.position      = data["direction"].as<glm::vec3>();
	spot_light->data.cut_off       = data["direction"].as<float>();
	spot_light->data.outer_cut_off = data["direction"].as<float>();
}

template <>
void deserialize_component<cmpt::Light>(Entity entity, const YAML::Node &data)
{
	if (!data)
	{
		return;
	}

	auto &light = entity.addComponent<cmpt::Light>();
	light.type  = static_cast<cmpt::LightType>(data["type"].as<uint32_t>());
	switch (light.type)
	{
		case cmpt::LightType::Directional:
			deserialize_component<cmpt::DirectionalLight>(entity, data);
			break;
		case cmpt::LightType::Point:
			deserialize_component<cmpt::PointLight>(entity, data);
			break;
		case cmpt::LightType::Spot:
			deserialize_component<cmpt::SpotLight>(entity, data);
			break;
		default:
			break;
	}
}

entt::entity deserialize_entity(const YAML::Node &data)
{
	auto entity = Scene::instance()->createEntity();

	deserialize_component<cmpt::Tag>(entity, data[typeid(cmpt::Tag).name()]);
	deserialize_component<cmpt::Transform>(entity, data[typeid(cmpt::Transform).name()]);
	deserialize_component<cmpt::Hierarchy>(entity, data[typeid(cmpt::Hierarchy).name()]);
	deserialize_component<cmpt::MeshRenderer>(entity, data[typeid(cmpt::MeshRenderer).name()]);
	deserialize_component<cmpt::Light>(entity, data[typeid(cmpt::Light).name()]);

	return entity;
}

void SceneSerializer::serialize(const std::string &file_path)
{
	YAML::Emitter emitter;
	emitter << YAML::BeginMap;

	// Scene
	emitter << YAML::Key << "Scene" << YAML::Value << Scene::instance()->name;

	// Main Camera
	emitter << YAML::Key << "Main Camera";
	serialize_main_camera(emitter);

	// Color correction
	emitter << YAML::Key << "Color Correction";
	serialize_color_correction(emitter);

	// Bloom
	emitter << YAML::Key << "Bloom";
	serialize_bloom(emitter);

	// Entities
	emitter << YAML::Key << "Entities" << YAML::Value << YAML::BeginSeq;
	Scene::instance()->getRegistry().each([&emitter](const entt::entity &entity) {
		serialize_entity(emitter, entity);
	});
	emitter << YAML::EndSeq;
	emitter << YAML::EndMap;

	std::ofstream fout(file_path);
	fout << emitter.c_str();
}

void SceneSerializer::deserialize(const std::string &file_path)
{
	YAML::Node data = YAML::LoadFile(file_path);

	load_images.clear();
	load_models.clear();

	if (!data["Scene"])
	{
		return;
	}

	Scene::instance()->clear();
	Renderer::instance()->getResourceCache().clear();

	// Scene
	Scene::instance()->name = data["Scene"].as<std::string>();

	// Main Camera
	deserialize_main_camera(data["Main Camera"]);

	// Color correction
	deserialize_color_correction(data["Color Correction"]);

	// Bloom
	deserialize_bloom(data["Bloom"]);

	// Entities
	if (!data)
	{
		return;
	}

	for (auto entity : data["Entities"])
	{
		m_entity_lut[entity["UUID"].as<uint32_t>()] = deserialize_entity(entity);
	}

	m_entity_lut[static_cast<uint32_t>(entt::null)] = entt::null;

	Scene::instance()->getRegistry().each([this](entt::entity entity) {
		auto &hierarchy  = Entity(entity).getComponent<cmpt::Hierarchy>();
		hierarchy.first  = m_entity_lut[static_cast<uint32_t>(hierarchy.first)];
		hierarchy.next   = m_entity_lut[static_cast<uint32_t>(hierarchy.next)];
		hierarchy.prev   = m_entity_lut[static_cast<uint32_t>(hierarchy.prev)];
		hierarchy.parent = m_entity_lut[static_cast<uint32_t>(hierarchy.parent)];
	});

	// Load resource
	for (auto &model : load_models)
	{
		Renderer::instance()->getResourceCache().loadModelAsync(model);
	}
	load_models.clear();
	for (auto &image : load_images)
	{
		if (FileSystem::isExist(image))
		{
			Renderer::instance()->getResourceCache().loadImageAsync(image);
		}
	}
	load_images.clear();
}
}        // namespace Ilum