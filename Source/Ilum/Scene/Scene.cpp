#include "Scene.hpp"
#include "Entity.hpp"

#include <Core/Path.hpp>

#include <RHI/Command.hpp>
#include <RHI/Texture.hpp>

#include <Asset/AssetManager.hpp>
#include <Asset/Material.hpp>

#include "Component/Camera.hpp"
#include "Component/Hierarchy.hpp"
#include "Component/MeshRenderer.hpp"
#include "Component/Tag.hpp"
#include "Component/Transform.hpp"

#include <cereal/archives/binary.hpp>
#include <cereal/archives/json.hpp>

#include <imgui.h>

#include <cgltf/cgltf.h>

#include <glm/gtc/type_ptr.hpp>
#include <glm/matrix.hpp>

#include <meshoptimizer.h>

#include <ImGuizmo.h>

#include <fstream>

namespace Ilum
{
inline void TransformRecursive(Scene &scene, entt::entity entity)
{
	if (entity == entt::null)
	{
		return;
	}

	auto &transform = Entity(scene, entity).GetComponent<cmpt::Transform>();
	auto &hierarchy = Entity(scene, entity).GetComponent<cmpt::Hierarchy>();
	if (hierarchy.parent != entt::null)
	{
		transform.local_transform = glm::scale(glm::translate(glm::mat4(1.f), transform.translation) * glm::mat4_cast(glm::qua<float>(glm::radians(transform.rotation))), transform.scale);
		transform.world_transform = Entity(scene, hierarchy.parent).GetComponent<cmpt::Transform>().world_transform * transform.local_transform;
	}

	auto child = hierarchy.first;

	while (child != entt::null)
	{
		TransformRecursive(scene, child);
		child = Entity(scene, child).GetComponent<cmpt::Hierarchy>().next;
	}
}

template <typename T>
inline std::string GetComponentName()
{
	std::string name   = typeid(T).name();
	size_t      offset = name.find_last_of(':');
	return name.substr(offset + 1, name.length());
}

// Draw Scene Hierarchy
inline void DeleteNode(Scene &scene, Entity entity)
{
	if (!entity)
	{
		return;
	}

	auto child   = Entity(scene, entity.GetComponent<cmpt::Hierarchy>().first);
	auto sibling = child ? Entity(scene, child.GetComponent<cmpt::Hierarchy>().next) : Entity(scene);
	while (sibling)
	{
		auto tmp = Entity(scene, sibling.GetComponent<cmpt::Hierarchy>().next);
		DeleteNode(scene, sibling);
		sibling = tmp;
	}
	DeleteNode(scene, child);
	entity.Destroy();
}

inline bool IsParentOf(Scene &scene, Entity lhs, Entity rhs)
{
	auto parent = Entity(scene, rhs.GetComponent<cmpt::Hierarchy>().parent);
	while (parent)
	{
		if (parent == lhs)
		{
			return true;
		}
		parent = Entity(scene, parent.GetComponent<cmpt::Hierarchy>().parent);
	}
	return false;
}

inline void SetAsSon(Scene &scene, Entity new_parent, Entity new_son)
{
	if (new_parent && IsParentOf(scene, new_son, new_parent))
	{
		return;
	}

	auto &h2 = new_son.GetComponent<cmpt::Hierarchy>();

	if (h2.next != entt::null)
	{
		Entity(scene, h2.next).GetComponent<cmpt::Hierarchy>().prev = h2.prev;
	}

	if (h2.prev != entt::null)
	{
		Entity(scene, h2.prev).GetComponent<cmpt::Hierarchy>().next = h2.next;
	}

	if (h2.parent != entt::null && new_son == Entity(scene, h2.parent).GetComponent<cmpt::Hierarchy>().first)
	{
		Entity(scene, h2.parent).GetComponent<cmpt::Hierarchy>().first = h2.next;
	}

	h2.next   = new_parent ? new_parent.GetComponent<cmpt::Hierarchy>().first : entt::null;
	h2.prev   = entt::null;
	h2.parent = new_parent ? new_parent.GetHandle() : entt::null;

	if (new_parent && new_parent.GetComponent<cmpt::Hierarchy>().first != entt::null)
	{
		Entity(scene, new_parent.GetComponent<cmpt::Hierarchy>().first).GetComponent<cmpt::Hierarchy>().prev = new_son;
	}

	if (new_parent)
	{
		new_parent.GetComponent<cmpt::Hierarchy>().first = new_son;
	}
}

inline void DrawNode(Scene &scene, Entity entity, Entity &select)
{
	if (!entity)
	{
		return;
	}

	auto &tag = entity.GetComponent<cmpt::Tag>().name;

	bool has_child = entity.HasComponent<cmpt::Hierarchy>() && entity.GetComponent<cmpt::Hierarchy>().first != entt::null;

	// Setting up
	ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(5, 5));
	ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_FramePadding | ImGuiTreeNodeFlags_DefaultOpen | (select == entity ? ImGuiTreeNodeFlags_Selected : 0) | (has_child ? 0 : ImGuiTreeNodeFlags_Leaf);

	bool open = ImGui::TreeNodeEx(std::to_string(entity).c_str(), flags, "%s", tag.c_str());

	ImGui::PopStyleVar();

	// Delete entity
	ImGui::PushID(std::to_string(entity).c_str());
	bool entity_deleted = false;
	if (ImGui::BeginPopupContextItem(std::to_string(entity).c_str()))
	{
		if (ImGui::MenuItem("Delete Entity"))
		{
			entity_deleted = true;
		}
		ImGui::EndPopup();
	}
	ImGui::PopID();

	// Select entity
	if (ImGui::IsItemClicked())
	{
		select = entity;
	}

	// Drag and drop
	if (ImGui::BeginDragDropSource())
	{
		if (entity.HasComponent<cmpt::Hierarchy>())
		{
			ImGui::SetDragDropPayload("Entity", &entity, sizeof(Entity));
		}
		ImGui::EndDragDropSource();
	}

	if (ImGui::BeginDragDropTarget())
	{
		if (const auto *pay_load = ImGui::AcceptDragDropPayload("Entity"))
		{
			ASSERT(pay_load->DataSize == sizeof(Entity));
			SetAsSon(scene, entity, *static_cast<Entity *>(pay_load->Data));
		}
		ImGui::EndDragDropTarget();
	}

	// Recursively open children entities
	if (open)
	{
		if (has_child)
		{
			auto child = Entity(scene, entity.GetComponent<cmpt::Hierarchy>().first);
			while (child)
			{
				DrawNode(scene, child, select);
				if (child)
				{
					child = Entity(scene, child.GetComponent<cmpt::Hierarchy>().next);
				}
			}
		}
		ImGui::TreePop();
	}

	// Delete callback
	if (entity_deleted)
	{
		if (Entity(scene, entity.GetComponent<cmpt::Hierarchy>().prev))
		{
			Entity(scene, entity.GetComponent<cmpt::Hierarchy>().prev).GetComponent<cmpt::Hierarchy>().next = entity.GetComponent<cmpt::Hierarchy>().next;
		}

		if (Entity(scene, entity.GetComponent<cmpt::Hierarchy>().next))
		{
			Entity(scene, entity.GetComponent<cmpt::Hierarchy>().next).GetComponent<cmpt::Hierarchy>().prev = entity.GetComponent<cmpt::Hierarchy>().prev;
		}

		if (Entity(scene, entity.GetComponent<cmpt::Hierarchy>().parent) && entity == Entity(scene, entity.GetComponent<cmpt::Hierarchy>().parent).GetComponent<cmpt::Hierarchy>().first)
		{
			Entity(scene, entity.GetComponent<cmpt::Hierarchy>().parent).GetComponent<cmpt::Hierarchy>().first = entity.GetComponent<cmpt::Hierarchy>().next;
		}

		DeleteNode(scene, entity);

		if (select == entity)
		{
			select = entity;
		}
	}
}

template <typename T, typename Callback>
inline bool DrawComponent(const std::string &name, Scene &scene, entt::entity entity, ImGuiContext &context, Callback callback, bool static_mode = false)
{
	const ImGuiTreeNodeFlags tree_node_flags = ImGuiTreeNodeFlags_DefaultOpen | ImGuiTreeNodeFlags_Framed | ImGuiTreeNodeFlags_SpanAvailWidth | ImGuiTreeNodeFlags_AllowItemOverlap | ImGuiTreeNodeFlags_FramePadding;

	Entity e(scene, entity);

	if (e.HasComponent<T>())
	{
		auto  &component                = e.GetComponent<T>();
		ImVec2 content_region_available = ImGui::GetContentRegionAvail();

		ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2{4, 4});
		float line_height = GImGui->Font->FontSize + GImGui->Style.FramePadding.y * 2.0f;
		bool  open        = ImGui::TreeNodeEx((void *) typeid(T).hash_code(), tree_node_flags, name.c_str());
		ImGui::PopStyleVar();

		bool remove_component = false;
		if (!static_mode)
		{
			ImGui::SameLine(content_region_available.x - line_height * 0.5f);
			if (ImGui::Button("-", ImVec2{line_height, line_height}))
			{
				remove_component = true;
			}
		}

		bool update = false;

		if (open)
		{
			update = callback(component, context);
			ImGui::TreePop();
		}

		if (remove_component)
		{
			e.RemoveComponent<T>();
			update = true;
		}

		return update;
	}

	return false;
}

template <typename Cmpt>
inline bool DrawComponent(Scene &scene, entt::entity entity, ImGuiContext &context, bool static_mode = false)
{
	static_assert(std::is_base_of_v<cmpt::Component, Cmpt>);
	return DrawComponent<Cmpt>(
	    GetComponentName<Cmpt>(), scene, entity, context, [](Cmpt &cmpt, ImGuiContext &context) {
		    return cmpt.OnImGui(context);
	    },
	    static_mode);
}

template <typename T>
inline void AddComponents(Scene &scene, entt::entity entity)
{
	Entity e(scene, entity);
	if (!e.HasComponent<T>() && ImGui::MenuItem(typeid(T).name()))
	{
		e.AddComponent<T>();
		ImGui::CloseCurrentPopup();
	}
}

Scene::Scene(RHIDevice *device, AssetManager &asset_manager, const std::string &name) :
    p_device(device), m_asset_manager(asset_manager), m_name(name)
{
	m_top_level_acceleration_structure = std::make_unique<AccelerationStructure>(p_device);

	m_main_camera_buffer = std::make_unique<Buffer>(
	    p_device,
	    BufferDesc(
	        sizeof(ShaderInterop::Camera),
	        1,
	        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
	        VMA_MEMORY_USAGE_CPU_TO_GPU));
}

entt::registry &Scene::GetRegistry()
{
	return m_registry;
}

AssetManager &Scene::GetAssetManager()
{
	return m_asset_manager;
}

const std::string &Scene::GetName() const
{
	return m_name;
}

const std::string &Scene::GetSavePath() const
{
	return m_save_path;
}

AccelerationStructure &Scene::GetTLAS()
{
	return *m_top_level_acceleration_structure;
}

Buffer &Scene::GetMainCameraBuffer()
{
	return *m_main_camera_buffer;
}

entt::entity Scene::GetMainCamera()
{
	return m_main_camera;
}

void Scene::Clear()
{
	m_registry.each([&](auto entity) { m_registry.destroy(entity); });
}

entt::entity Scene::Create(const std::string &name)
{
	auto entity = m_registry.create();
	m_registry.emplace<cmpt::Tag>(entity, name);
	m_registry.emplace<cmpt::Transform>(entity);
	m_registry.emplace<cmpt::Hierarchy>(entity);
	return entity;
}

void Scene::Tick()
{
	// Update camera
	if (m_registry.valid(m_main_camera))
	{
		ShaderInterop::Camera *camera_data = static_cast<ShaderInterop::Camera *>(m_main_camera_buffer->Map());

		Entity entity = Entity(*this, m_main_camera);

		auto &camera    = entity.GetComponent<cmpt::Camera>();
		auto &transform = entity.GetComponent<cmpt::Transform>();

		camera.view            = glm::inverse(transform.world_transform);
		camera.projection      = camera.type == cmpt::CameraType::Perspective ?
		                             glm::perspective(glm::radians(camera.fov), camera.aspect, camera.near_plane, camera.far_plane) :
                                     glm::ortho(camera.left, camera.right, camera.bottom, camera.top, camera.near_plane, camera.far_plane);
		camera.view_projection = camera.projection * camera.view;
		camera.frame_count++;

		camera_data->view            = camera.view;
		camera_data->projection      = camera.projection;
		camera_data->inv_view        = transform.world_transform;
		camera_data->inv_projection  = glm::inverse(camera.projection);
		camera_data->view_projection = camera.projection * camera.view;
		camera_data->position        = transform.translation;
		camera_data->frame_count     = camera.frame_count;

		m_main_camera_buffer->Flush(m_main_camera_buffer->GetSize());
		m_main_camera_buffer->Unmap();
	}

	// Update scene
	if (m_update)
	{
		// Update Transform
		auto group = m_registry.group<>(entt::get<cmpt::Transform, cmpt::Hierarchy>);

		std::vector<entt::entity> roots;

		// Find roots
		for (auto &entity : group)
		{
			auto &&[hierarchy, transform] = group.get<cmpt::Hierarchy, cmpt::Transform>(entity);

			if (hierarchy.parent == entt::null)
			{
				transform.local_transform = glm::scale(glm::translate(glm::mat4(1.f), transform.translation) * glm::mat4_cast(glm::qua<float>(glm::radians(transform.rotation))), transform.scale);
				transform.world_transform = transform.local_transform;
				roots.emplace_back(entity);
			}
		}

		for (auto &root : roots)
		{
			TransformRecursive(*this, root);
		}

		// Update buffer
		std::vector<VkAccelerationStructureInstanceKHR> acceleration_structure_instances;
		acceleration_structure_instances.reserve(100);
		auto mesh_view = m_registry.view<cmpt::MeshRenderer>();
		for (size_t i = 0; i < mesh_view.size(); i++)
		{
			Entity entity        = Entity(*this, mesh_view[i]);
			auto  &mesh_renderer = entity.GetComponent<cmpt::MeshRenderer>();
			auto  &transform     = entity.GetComponent<cmpt::Transform>();

			if (!mesh_renderer.buffer)
			{
				BufferDesc desc      = {};
				desc.size            = sizeof(ShaderInterop::Instance);
				desc.buffer_usage    = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
				desc.memory_usage    = VMA_MEMORY_USAGE_CPU_TO_GPU;
				mesh_renderer.buffer = std::make_unique<Buffer>(p_device, desc);
			}

			auto *instance_data      = static_cast<ShaderInterop::Instance *>(mesh_renderer.buffer->Map());
			instance_data->transform = transform.world_transform;
			instance_data->material  = m_asset_manager.GetIndex(mesh_renderer.mesh->GetMaterial());
			instance_data->mesh      = m_asset_manager.GetIndex(mesh_renderer.mesh);
			instance_data->meshlet_count = mesh_renderer.mesh->GetMeshletsCount();
			mesh_renderer.buffer->Flush(mesh_renderer.buffer->GetSize());
			mesh_renderer.buffer->Unmap();

			VkAccelerationStructureInstanceKHR acceleration_structure_instance = {};

			auto transform_transpose = glm::mat3x4(glm::transpose(transform.world_transform));
			std::memcpy(&acceleration_structure_instance.transform, &transform_transpose, sizeof(VkTransformMatrixKHR));
			acceleration_structure_instance.instanceCustomIndex                    = 0;
			acceleration_structure_instance.mask                                   = 0xFF;
			acceleration_structure_instance.instanceShaderBindingTableRecordOffset = 0;
			acceleration_structure_instance.flags                                  = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
			acceleration_structure_instance.accelerationStructureReference         = mesh_renderer.mesh->GetBLAS().GetDeviceAddress();
			acceleration_structure_instances.push_back(acceleration_structure_instance);
		}

		if (!acceleration_structure_instances.empty())
		{
			// Update TLAS
			Buffer as_instance_buffer(
			    p_device,
			    BufferDesc(
			        sizeof(VkAccelerationStructureInstanceKHR),
			        static_cast<uint32_t>(acceleration_structure_instances.size()),
			        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
			            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
			            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
			        VMA_MEMORY_USAGE_CPU_TO_GPU));
			std::memcpy(as_instance_buffer.Map(), acceleration_structure_instances.data(), as_instance_buffer.GetSize());
			as_instance_buffer.Flush(as_instance_buffer.GetSize());
			as_instance_buffer.Unmap();

			AccelerationStructureDesc as_desc                      = {};
			as_desc.type                                           = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
			as_desc.geometry.sType                                 = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
			as_desc.geometry.geometryType                          = VK_GEOMETRY_TYPE_INSTANCES_KHR;
			as_desc.geometry.flags                                 = VK_GEOMETRY_OPAQUE_BIT_KHR;
			as_desc.geometry.geometry.instances.sType              = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
			as_desc.geometry.geometry.instances.arrayOfPointers    = VK_FALSE;
			as_desc.geometry.geometry.instances.data.deviceAddress = as_instance_buffer.GetDeviceAddress();
			as_desc.range_info.primitiveCount                      = static_cast<uint32_t>(acceleration_structure_instances.size());
			as_desc.range_info.primitiveOffset                     = 0;
			as_desc.range_info.firstVertex                         = 0;
			as_desc.range_info.transformOffset                     = 0;

			{
				auto &cmd_buffer = p_device->RequestCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, VK_QUEUE_COMPUTE_BIT);
				cmd_buffer.Begin();
				m_top_level_acceleration_structure->Build(cmd_buffer, as_desc);
				cmd_buffer.End();
				p_device->SubmitIdle(cmd_buffer, VK_QUEUE_COMPUTE_BIT);
			}
		}
		m_update = false;
	}
}

void Scene::OnImGui(ImGuiContext &context)
{
	if (ImGui::BeginMainMenuBar())
	{
		if (ImGui::BeginMenu("Main Camera"))
		{
			auto camera_view = m_registry.view<cmpt::Camera>();
			for (auto &camera : camera_view)
			{
				bool select_camera = (camera == m_main_camera);
				if (ImGui::MenuItem(Entity(*this, camera).GetComponent<cmpt::Tag>().name.c_str(), nullptr, &select_camera))
				{
					if (select_camera)
					{
						m_main_camera = camera;
					}
					else
					{
						m_main_camera = entt::null;
					}
				}
			}
			ImGui::EndMenu();
		}
		ImGui::EndMainMenuBar();
	}

	// Scene Hierarchy
	{
		ImGui::Begin("Scene Hierarchy");

		{
			char buffer[64];
			std::memset(buffer, 0, sizeof(buffer));
			std::memcpy(buffer, m_name.data(), sizeof(buffer));
			ImGui::PushItemWidth(150.f);
			if (ImGui::InputText("##SceneName", buffer, sizeof(buffer)))
			{
				m_name = std::string(buffer);
			}
		}

		if (ImGui::BeginDragDropTarget())
		{
			if (const auto *pay_load = ImGui::AcceptDragDropPayload("Entity"))
			{
				ASSERT(pay_load->DataSize == sizeof(Entity));
				SetAsSon(*this, Entity(*this), *static_cast<Entity *>(pay_load->Data));
				m_update = true;
			}
		}

		Entity select_entity(*this, m_select);
		m_registry.each([this, &select_entity](auto entity_id) {
			if (Entity(*this, entity_id) && Entity(*this, entity_id).GetComponent<cmpt::Hierarchy>().parent == entt::null)
			{
				DrawNode(*this, Entity(*this, entity_id), select_entity);
			}
		});
		m_select = select_entity;

		if (ImGui::IsMouseDown(ImGuiMouseButton_Left) && ImGui::IsWindowHovered())
		{
			m_select = entt::null;
		}

		// Right-click on blank space
		if (ImGui::BeginPopupContextWindow(0, 1, false))
		{
			if (ImGui::MenuItem("New Entity"))
			{
				Create("Untitled Entity");
			}
			ImGui::EndPopup();
		}
		ImGui::End();
	}

	// Entity Inspector
	ImGui::Begin("Entity Inspector");
	if (m_registry.valid(m_select))
	{
		Entity(*this, m_select).GetComponent<cmpt::Tag>().OnImGui(context);
		ImGui::SameLine();
		ImGui::PushItemWidth(-1);

		if (ImGui::Button("Add Component"))
		{
			ImGui::OpenPopup("AddComponent");
		}
		ImGui::PopItemWidth();

		if (ImGui::BeginPopup("AddComponent"))
		{
			AddComponents<cmpt::MeshRenderer>(*this, m_select);
			AddComponents<cmpt::Camera>(*this, m_select);

			ImGui::EndPopup();
		}

		m_update |= DrawComponent<cmpt::Transform>(*this, m_select, context, true);
		m_update |= DrawComponent<cmpt::Hierarchy>(*this, m_select, context, true);
		m_update |= DrawComponent<cmpt::MeshRenderer>(*this, m_select, context, false);
		DrawComponent<cmpt::Camera>(*this, m_select, context, false);
	}
	ImGui::End();
}

void Scene::Save(const std::string &filename)
{
	m_save_path = filename;
	m_name      = Path::GetInstance().GetFileName(filename, false);

	std::ofstream os(filename, std::ios::binary);

	cereal::JSONOutputArchive archive(os);

	entt::snapshot{m_registry}
	    .entities(archive)
	    .component<
	        cmpt::Tag,
	        cmpt::Transform,
	        cmpt::Hierarchy>(archive);
}

void Scene::Load(const std::string &filename)
{
	std::ifstream os(filename.empty() ? m_save_path : filename, std::ios::binary);

	m_save_path = filename;
	m_name      = Path::GetInstance().GetFileName(filename, false);

	cereal::JSONInputArchive archive(os);

	Clear();

	entt::snapshot_loader{m_registry}
	    .entities(archive)
	    .component<
	        cmpt::Tag,
	        cmpt::Transform,
	        cmpt::Hierarchy>(archive);
}

void Scene::ImportGLTF(const std::string &filename)
{
	cgltf_options options  = {};
	cgltf_data   *raw_data = nullptr;
	cgltf_result  result   = cgltf_parse_file(&options, filename.c_str(), &raw_data);
	if (result != cgltf_result_success)
	{
		LOG_WARN("Failed to load gltf {}", filename);
		return;
	}
	result = cgltf_load_buffers(&options, raw_data, filename.c_str());
	if (result != cgltf_result_success)
	{
		LOG_WARN("Failed to load gltf {}", filename);
		return;
	}

	std::map<cgltf_texture *, Texture *> texture_map;

	auto load_texture = [this, &texture_map, filename](Texture *&texture, cgltf_texture *gltf_texture) {
		if (!gltf_texture)
		{
			return;
		}
		if (texture_map.find(gltf_texture) != texture_map.end())
		{
			texture = texture_map[gltf_texture];
			return;
		}
		if (gltf_texture->image->uri)
		{
			texture = m_asset_manager.LoadTexture(Path::GetInstance().GetFileDirectory(filename) + gltf_texture->image->uri);
		}
		else if (gltf_texture->image->buffer_view)
		{
			auto new_texture = std::make_unique<Texture>(
			    p_device,
			    static_cast<char *>(gltf_texture->image->buffer_view->buffer->data) + gltf_texture->image->buffer_view->offset,
			    static_cast<int32_t>(gltf_texture->image->buffer_view->size));
			if (gltf_texture->image->name)
			{
				new_texture->SetName(gltf_texture->image->name);
			}
			texture = m_asset_manager.Add(std::move(new_texture));
		}
	};

	std::map<cgltf_material *, Material *> material_map;

	// Load materials
	for (uint32_t i = 0; i < raw_data->materials_count; i++)
	{
		auto &raw_material          = raw_data->materials[i];
		auto *material              = m_asset_manager.Add(std::make_unique<Material>(p_device, m_asset_manager));
		material_map[&raw_material] = material;
		if (raw_material.name)
		{
			material->m_name = raw_material.name;
		}
		else
		{
			material->m_name = Path::GetInstance().GetFileName(filename, false) + " Material #" + std::to_string(i);
		}
		material->m_alpha_mode        = static_cast<AlphaMode>(raw_material.alpha_mode);
		material->m_alpha_cut_off     = raw_material.alpha_cutoff;
		material->m_emissive_strength = raw_material.emissive_strength.emissive_strength;
		load_texture(material->m_normal_texture, raw_material.normal_texture.texture);
		load_texture(material->m_emissive_texture, raw_material.emissive_texture.texture);
		std::memcpy(glm::value_ptr(material->m_emissive_factor), raw_material.emissive_factor, sizeof(material->m_emissive_factor));
		if (raw_material.has_pbr_metallic_roughness)
		{
			std::memcpy(glm::value_ptr(material->m_albedo_factor), raw_material.pbr_metallic_roughness.base_color_factor, sizeof(material->m_albedo_factor));
			material->m_metallic_factor  = raw_material.pbr_metallic_roughness.metallic_factor;
			material->m_roughness_factor = raw_material.pbr_metallic_roughness.roughness_factor;
			load_texture(material->m_metallic_roughness_texture, raw_material.pbr_metallic_roughness.metallic_roughness_texture.texture);
			load_texture(material->m_albedo_texture, raw_material.pbr_metallic_roughness.base_color_texture.texture);
		}
		else if (raw_material.has_pbr_specular_glossiness)
		{
			std::memcpy(glm::value_ptr(material->m_albedo_factor), raw_material.pbr_specular_glossiness.diffuse_factor, sizeof(material->m_albedo_factor));
			std::memcpy(glm::value_ptr(material->m_specular_factor), raw_material.pbr_specular_glossiness.specular_factor, sizeof(material->m_specular_factor));
			material->m_glossiness_factor = raw_material.pbr_specular_glossiness.glossiness_factor;
			load_texture(material->m_albedo_texture, raw_material.pbr_specular_glossiness.diffuse_texture.texture);
			load_texture(material->m_specular_glossiness_texture, raw_material.pbr_specular_glossiness.specular_glossiness_texture.texture);
			load_texture(material->m_albedo_texture, raw_material.pbr_specular_glossiness.diffuse_texture.texture);
		}
	}

	std::map<cgltf_mesh *, std::vector<Mesh *>> mesh_map;

	// Load mesh
	for (uint32_t mesh_id = 0; mesh_id < raw_data->meshes_count; mesh_id++)
	{
		auto       &raw_mesh  = raw_data->meshes[mesh_id];
		std::string mesh_name = raw_mesh.name ? raw_mesh.name : Path::GetInstance().GetFileName(filename, false) + " Mesh #" + std::to_string(mesh_id);

		for (uint32_t prim_id = 0; prim_id < raw_mesh.primitives_count; prim_id++)
		{
			const cgltf_primitive &primitive = raw_mesh.primitives[prim_id];

			std::unique_ptr<Mesh> submesh = std::make_unique<Mesh>(p_device, m_asset_manager);
			submesh->m_material           = material_map[primitive.material];
			submesh->m_name               = mesh_name + " SubMesh #" + std::to_string(prim_id);

			submesh->m_indices.resize(primitive.indices->count);
			for (size_t i = 0; i < primitive.indices->count; i += 3)
			{
				submesh->m_indices[i + 0] = static_cast<uint32_t>(cgltf_accessor_read_index(primitive.indices, i + 0));
				submesh->m_indices[i + 1] = static_cast<uint32_t>(cgltf_accessor_read_index(primitive.indices, i + 2));
				submesh->m_indices[i + 2] = static_cast<uint32_t>(cgltf_accessor_read_index(primitive.indices, i + 1));
			}
			for (size_t attr_id = 0; attr_id < primitive.attributes_count; attr_id++)
			{
				const cgltf_attribute &attribute = primitive.attributes[attr_id];
				const char            *attr_name = attribute.name;
				if (strcmp(attr_name, "POSITION") == 0)
				{
					submesh->m_vertices.resize(attribute.data->count);
					for (size_t i = 0; i < attribute.data->count; ++i)
					{
						cgltf_accessor_read_float(attribute.data, i, &submesh->m_vertices[i].position.x, 3);
						submesh->m_bounding_box.Merge(submesh->m_vertices[i].position);
					}
				}
				else if (strcmp(attr_name, "NORMAL") == 0)
				{
					submesh->m_vertices.resize(attribute.data->count);
					for (size_t i = 0; i < attribute.data->count; ++i)
					{
						cgltf_accessor_read_float(attribute.data, i, &submesh->m_vertices[i].normal.x, 3);
					}
				}
				else if (strcmp(attr_name, "TANGENT") == 0)
				{
					submesh->m_vertices.resize(attribute.data->count);
					for (size_t i = 0; i < attribute.data->count; ++i)
					{
						cgltf_accessor_read_float(attribute.data, i, &submesh->m_vertices[i].tangent.x, 3);
					}
				}
				else if (strcmp(attr_name, "TEXCOORD_0") == 0)
				{
					submesh->m_vertices.resize(attribute.data->count);
					for (size_t i = 0; i < attribute.data->count; ++i)
					{
						cgltf_accessor_read_float(attribute.data, i, &submesh->m_vertices[i].texcoord.x, 2);
					}
				}
				else
				{
					LOG_WARN("GLTF - Attribute {} is unsupported", attr_name);
				}
			}
			//	Optimize mesh
			meshopt_optimizeVertexCache(submesh->m_indices.data(), submesh->m_indices.data(), submesh->m_indices.size(), submesh->m_vertices.size());
			meshopt_optimizeOverdraw(submesh->m_indices.data(), submesh->m_indices.data(), submesh->m_indices.size(), &submesh->m_vertices[0].position.x, submesh->m_vertices.size(), sizeof(ShaderInterop::Vertex), 1.05f);
			std::vector<uint32_t> remap(submesh->m_vertices.size());
			meshopt_optimizeVertexFetchRemap(&remap[0], submesh->m_indices.data(), submesh->m_indices.size(), submesh->m_vertices.size());
			meshopt_remapIndexBuffer(submesh->m_indices.data(), submesh->m_indices.data(), submesh->m_indices.size(), &remap[0]);
			meshopt_remapVertexBuffer(submesh->m_vertices.data(), submesh->m_vertices.data(), submesh->m_vertices.size(), sizeof(ShaderInterop::Vertex), &remap[0]);

			// Generate meshlet
			const size_t max_vertices  = ShaderInterop::MESHLET_MAX_VERTICES;
			const size_t max_triangles = ShaderInterop::MESHLET_MAX_TRIANGLES;
			const float  cone_weight   = 0.5f;

			size_t max_meshlets = meshopt_buildMeshletsBound(submesh->m_indices.size(), max_vertices, max_triangles);

			submesh->m_meshlets.resize(max_meshlets);
			submesh->m_meshlet_vertices.resize(max_meshlets * max_vertices);

			std::vector<uint8_t>         meshlet_triangles(max_meshlets * max_triangles * 3);
			std::vector<meshopt_Meshlet> meshlets(max_meshlets);

			size_t meshlet_count = meshopt_buildMeshlets(meshlets.data(), submesh->m_meshlet_vertices.data(), meshlet_triangles.data(),
			                                             submesh->m_indices.data(), submesh->m_indices.size(), &submesh->m_vertices[0].position.x, submesh->m_vertices.size(),
			                                             sizeof(ShaderInterop::Vertex), max_vertices, max_triangles, cone_weight);

			const meshopt_Meshlet &last = meshlets[meshlet_count - 1];
			meshlet_triangles.resize(last.triangle_offset + ((last.triangle_count * 3 + 3) & ~3));
			meshlets.resize(meshlet_count);

			submesh->m_meshlet_vertices.resize(last.vertex_offset + last.vertex_count);
			submesh->m_meshlets.resize(meshlet_count);
			submesh->m_meshlet_triangles.resize(meshlet_triangles.size());

			uint32_t triangle_offset = 0;
			for (size_t i = 0; i < meshlet_count; i++)
			{
				const auto &meshlet = meshlets[i];
				const auto  bound   = meshopt_computeMeshletBounds(&submesh->m_meshlet_vertices[meshlet.vertex_offset], &meshlet_triangles[meshlet.triangle_offset],
				                                                   meshlet.triangle_count, &submesh->m_vertices[0].position.x, submesh->m_vertices.size(), sizeof(ShaderInterop::Vertex));

				submesh->m_meshlets[i].bound.center       = glm::vec3(bound.center[0], bound.center[1], bound.center[2]);
				submesh->m_meshlets[i].bound.radius       = bound.radius;
				submesh->m_meshlets[i].bound.cone_axis    = glm::vec3(bound.cone_axis[0], bound.cone_axis[1], bound.cone_axis[2]);
				submesh->m_meshlets[i].bound.cone_cut_off = bound.cone_cutoff;

				for (uint32_t tri_idx = 0; tri_idx < meshlet.triangle_count; tri_idx++)
				{
					submesh->m_meshlet_triangles[tri_idx + triangle_offset] = ShaderInterop::PackTriangle(
					    meshlet_triangles[tri_idx * 3 + 0 + meshlet.triangle_offset],
					    meshlet_triangles[tri_idx * 3 + 1 + meshlet.triangle_offset],
					    meshlet_triangles[tri_idx * 3 + 2 + meshlet.triangle_offset]);
				}
				submesh->m_meshlets[i].primitive_count  = meshlet.triangle_count;
				submesh->m_meshlets[i].primitive_offset = triangle_offset;
				submesh->m_meshlets[i].vertex_offset    = meshlet.vertex_offset;
				submesh->m_meshlets[i].vertex_count     = meshlet.vertex_count;
				triangle_offset += meshlet.triangle_count * 3;
			}
			submesh->m_meshlet_triangles.resize(triangle_offset);
			submesh->UpdateBuffer();
			mesh_map[&raw_mesh].push_back(m_asset_manager.Add(std::move(submesh)));
		}
	}

	// Load node
	std::unordered_map<const cgltf_node *, entt::entity> node_map;

	auto root = Entity(*this, Create(Path::GetInstance().GetFileName(filename, false).c_str()));

	for (size_t i = 0; i < raw_data->nodes_count; i++)
	{
		const cgltf_node &node = raw_data->nodes[i];

		std::string node_name = node.name ? node.name : (Path::GetInstance().GetFileName(filename, false) + std::string(" Node #") + std::to_string(i));
		auto        entity    = Entity(*this, Create(node_name.c_str()));

		SetAsSon(*this, root, entity);
		node_map[&node] = entity;

		auto &transform       = entity.GetComponent<cmpt::Transform>();
		transform.translation = glm::vec3(node.translation[0], node.translation[1], node.translation[2]);
		transform.rotation    = glm::vec3(node.rotation[0], node.rotation[1], node.rotation[2]);
		transform.scale       = glm::vec3(node.scale[0], node.scale[1], node.scale[2]);

		if (node.mesh)
		{
			uint32_t mesh_id = 0;
			for (auto &submesh : mesh_map[node.mesh])
			{
				auto  mesh_entity     = Entity(*this, Create((node_name + std::string(" Mesh #") + std::to_string(mesh_id++)).c_str()));
				auto &mesh_renderer   = mesh_entity.AddComponent<cmpt::MeshRenderer>();
				mesh_renderer.manager = &m_asset_manager;
				mesh_renderer.mesh    = submesh;
				SetAsSon(*this, entity, mesh_entity);
			}
		}
	}
	for (auto &[node, entity] : node_map)
	{
		if (node->parent)
		{
			SetAsSon(*this, Entity(*this, node_map[node->parent]), Entity(*this, entity));
		}
		for (size_t i = 0; i < node->children_count; i++)
		{
			SetAsSon(*this, Entity(*this, entity), Entity(*this, node_map[node->children[i]]));
		}
	}
	cgltf_free(raw_data);

	m_update = true;
}

void Scene::ExportGLTF(const std::string &filename)
{
}

}        // namespace Ilum