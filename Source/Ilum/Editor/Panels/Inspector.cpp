#include "Inspector.hpp"

#include "Editor/Editor.hpp"

#include "Scene/Component/Hierarchy.hpp"
#include "Scene/Component/Light.hpp"
#include "Scene/Component/Renderable.hpp"
#include "Scene/Component/Tag.hpp"
#include "Scene/Component/Transform.hpp"
#include "Scene/Component/Camera.hpp"

#include "Geometry/Shape/Sphere.hpp"

#include "Material/DisneyPBR.h"
#include "Material/Material.h"

#include "Renderer/Renderer.hpp"

#include "Loader/ImageLoader/ImageLoader.hpp"
#include "Loader/ResourceCache.hpp"

#include "ImGui/ImGuiContext.hpp"

#include "File/FileSystem.hpp"

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <imgui.h>
#include <imgui_internal.h>

namespace Ilum::panel
{
inline bool draw_vec3_control(const std::string &label, glm::vec3 &values, float resetValue = 0.0f, float columnWidth = 100.0f)
{
	ImGuiIO &io        = ImGui::GetIO();
	auto     bold_font = io.Fonts->Fonts[0];
	bool     update    = false;

	ImGui::PushID(label.c_str());

	ImGui::Columns(2);
	ImGui::SetColumnWidth(0, columnWidth);
	ImGui::Text(label.c_str());
	ImGui::NextColumn();

	ImGui::PushMultiItemsWidths(3, ImGui::CalcItemWidth());
	ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2{0, 0});

	float  line_height = GImGui->Font->FontSize + GImGui->Style.FramePadding.y * 2.0f;
	ImVec2 button_size = {line_height + 3.0f, line_height};

	ImGui::PushStyleColor(ImGuiCol_Button, ImVec4{0.8f, 0.1f, 0.15f, 1.0f});
	ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4{0.9f, 0.2f, 0.2f, 1.0f});
	ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4{0.8f, 0.1f, 0.15f, 1.0f});
	ImGui::PushFont(bold_font);
	if (ImGui::Button("X", button_size))
	{
		values.x = resetValue;
		update   = true;
	}
	ImGui::PopFont();
	ImGui::PopStyleColor(3);

	ImGui::SameLine();
	update = ImGui::DragFloat("##X", &values.x, 0.1f, 0.0f, 0.0f, "%.3f") || update;
	ImGui::PopItemWidth();
	ImGui::SameLine();

	ImGui::PushStyleColor(ImGuiCol_Button, ImVec4{0.2f, 0.7f, 0.2f, 1.0f});
	ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4{0.3f, 0.8f, 0.3f, 1.0f});
	ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4{0.2f, 0.7f, 0.2f, 1.0f});
	ImGui::PushFont(bold_font);
	if (ImGui::Button("Y", button_size))
	{
		values.y = resetValue;
		update   = true;
	}
	ImGui::PopFont();
	ImGui::PopStyleColor(3);

	ImGui::SameLine();
	update = ImGui::DragFloat("##Y", &values.y, 0.1f, 0.0f, 0.0f, "%.3f") || update;
	ImGui::PopItemWidth();
	ImGui::SameLine();

	ImGui::PushStyleColor(ImGuiCol_Button, ImVec4{0.1f, 0.25f, 0.8f, 1.0f});
	ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4{0.2f, 0.35f, 0.9f, 1.0f});
	ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4{0.1f, 0.25f, 0.8f, 1.0f});
	ImGui::PushFont(bold_font);
	if (ImGui::Button("Z", button_size))
	{
		values.z = resetValue;
		update   = true;
	}
	ImGui::PopFont();
	ImGui::PopStyleColor(3);

	ImGui::SameLine();
	update = ImGui::DragFloat("##Z", &values.z, 0.1f, 0.0f, 0.0f, "%.3f") || update;
	ImGui::PopItemWidth();

	ImGui::PopStyleVar();

	ImGui::Columns(1);

	ImGui::PopID();

	return update;
}

template <typename T>
void select_material(scope<IMaterial> &material)
{
	if ((!material || (material && material->type() != typeid(T))) && ImGui::MenuItem(typeid(T).name()))
	{
		material = createScope<T>();
	}
}

template <typename T1, typename T2, typename... Tn>
inline void select_material()
{
	select_material<T1>();
	select_material<T2, Tn...>();
}

template <typename T, typename Callback>
inline void draw_component(const std::string &name, Entity entity, Callback callback, bool static_mode = false)
{
	const ImGuiTreeNodeFlags tree_node_flags = ImGuiTreeNodeFlags_DefaultOpen | ImGuiTreeNodeFlags_Framed | ImGuiTreeNodeFlags_SpanAvailWidth | ImGuiTreeNodeFlags_AllowItemOverlap | ImGuiTreeNodeFlags_FramePadding;
	if (entity.hasComponent<T>())
	{
		auto & component                = entity.getComponent<T>();
		ImVec2 content_region_available = ImGui::GetContentRegionAvail();

		ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2{4, 4});
		float line_height = GImGui->Font->FontSize + GImGui->Style.FramePadding.y * 2.0f;
		ImGui::Separator();
		bool open = ImGui::TreeNodeEx((void *) typeid(T).hash_code(), tree_node_flags, name.c_str());
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

		if (open)
		{
			callback(component);
			ImGui::TreePop();
		}

		if (remove_component)
		{
			entity.removeComponent<T>();
		}
	}
}

template <typename T>
inline void draw_material(T &material)
{
	ASSERT(false);
}

void draw_texture(std::string &texture)
{
	if (ImGui::ImageButton(Renderer::instance()->getResourceCache().hasImage(FileSystem::getRelativePath(texture)) ?
                               ImGuiContext::textureID(Renderer::instance()->getResourceCache().loadImage(FileSystem::getRelativePath(texture)), Renderer::instance()->getSampler(Renderer::SamplerType::Trilinear_Clamp)) :
                               ImGuiContext::textureID(Renderer::instance()->getDefaultTexture(), Renderer::instance()->getSampler(Renderer::SamplerType::Trilinear_Clamp)),
	                       ImVec2{100.f, 100.f}))
	{
		texture = "";
	}

	if (ImGui::BeginDragDropTarget())
	{
		if (const auto *pay_load = ImGui::AcceptDragDropPayload("Texture2D"))
		{
			ASSERT(pay_load->DataSize == sizeof(std::string));
			if (texture != *static_cast<std::string *>(pay_load->Data))
			{
				texture = *static_cast<std::string *>(pay_load->Data);
			}
		}
		ImGui::EndDragDropTarget();
	}
}

template <>
inline void draw_material<material::DisneyPBR>(material::DisneyPBR &material)
{
	cmpt::Renderable::update = cmpt::Renderable::update || ImGui::ColorEdit4("Base Color", glm::value_ptr(material.base_color));
	cmpt::Renderable::update = cmpt::Renderable::update || ImGui::ColorEdit3("Emissive Color", glm::value_ptr(material.emissive_color));
	cmpt::Renderable::update = cmpt::Renderable::update || ImGui::DragFloat("Metallic Factor", &material.metallic_factor, 0.01f, 0.f, 1.f, "%.3f");
	cmpt::Renderable::update = cmpt::Renderable::update || ImGui::DragFloat("Emissive Intensity", &material.emissive_intensity, 0.01f, 0.f, std::numeric_limits<float>::max(), "%.3f");
	cmpt::Renderable::update = cmpt::Renderable::update || ImGui::DragFloat("Roughness Factor", &material.roughness_factor, 0.01f, 0.f, 1.f, "%.3f");
	cmpt::Renderable::update = cmpt::Renderable::update || ImGui::DragFloat("Height Factor", &material.displacement_height, 0.01f, 0.f, std::numeric_limits<float>::max(), "%.3f");

	ImGui::Text("Albedo Map");
	draw_texture(material.albedo_map);

	ImGui::Text("Normal Map");
	draw_texture(material.normal_map);

	ImGui::Text("Metallic Map");
	draw_texture(material.metallic_map);

	ImGui::Text("Roughness Map");
	draw_texture(material.roughness_map);

	ImGui::Text("Emissive Map");
	draw_texture(material.emissive_map);

	ImGui::Text("AO Map");
	draw_texture(material.ao_map);

	ImGui::Text("Displacement Map");
	draw_texture(material.displacement_map);
}

template <typename T>
inline void draw_material(scope<IMaterial> &material)
{
	if (material->type() == typeid(T))
	{
		draw_material<T>(*static_cast<T *>(material.get()));
	}
}

template <typename T1, typename T2, typename... Tn>
inline void draw_material(scope<IMaterial> &material)
{
	draw_material<T1>(material);
	draw_material<T2, Tn...>(material);
}

inline void draw_material(scope<IMaterial> &material)
{
	if (!material)
	{
		return;
	}

	draw_material<material::DisneyPBR>(material);
}

template <typename T>
inline void add_components()
{
	if (!Editor::instance()->getSelect().hasComponent<T>() && ImGui::MenuItem(typeid(T).name()))
	{
		Editor::instance()->getSelect().addComponent<T>();
		ImGui::CloseCurrentPopup();
	}
}

template <typename T1, typename T2, typename... Tn>
inline void add_components()
{
	add_components<T1>();
	add_components<T2, Tn...>();
}

template <typename Base, typename T>
inline bool has_component()
{
	return std::is_base_of_v<Base, T> && Editor::instance()->getSelect().hasComponent<T>();
}

template <typename Base, typename T1, typename T2, typename... Tn>
inline bool has_component()
{
	return has_component<Base, T1>() || has_component<Base, T2, Tn...>();
}

template <typename Base, typename T1, typename... Tn>
inline void add_component()
{
	if (has_component<Base, T1, Tn...>())
	{
		return;
	}

	if (ImGui::BeginMenu(typeid(Base).name()))
	{
		add_components<T1, Tn...>();
		ImGui::EndMenu();
	}
}

template <typename T>
inline void draw_component(Entity entity)
{
}

template <typename T1, typename T2, typename... Tn>
inline void draw_component(Entity entity)
{
	draw_component<T1>(entity);
	draw_component<T2, Tn...>(entity);
}

template <>
inline void draw_component<cmpt::Tag>(Entity entity)
{
	if (entity.hasComponent<cmpt::Tag>())
	{
		ImGui::Text("Tag");

		ImGui::SameLine();
		auto &tag = entity.getComponent<cmpt::Tag>().name;
		char  buffer[64];
		memset(buffer, 0, sizeof(buffer));
		std::memcpy(buffer, tag.data(), sizeof(buffer));
		ImGui::PushItemWidth(150.f);
		if (ImGui::InputText("##Tag", buffer, sizeof(buffer)))
		{
			tag = std::string(buffer);
		}
		ImGui::PopItemWidth();
		ImGui::SameLine();
		if (ImGui::Checkbox("Active", &entity.getComponent<cmpt::Tag>().active))
		{
			cmpt::Tag::update = true;
		}
	}
}

template <>
inline void draw_component<cmpt::Transform>(Entity entity)
{
	draw_component<cmpt::Transform>(
	    "Transform", entity, [](cmpt::Transform &component) {
		    component.update = draw_vec3_control("Translation", component.translation, 0.f) || component.update;
		    component.update = draw_vec3_control("Rotation", component.rotation, 0.f) || component.update;
		    component.update = draw_vec3_control("Scale", component.scale, 1.f) || component.update;
	    },
	    true);
}

template <>
inline void draw_component<cmpt::Hierarchy>(Entity entity)
{
	draw_component<cmpt::Hierarchy>(
	    "Hierarchy", entity, [](auto &component) {
		    ImGui::Text("Parent: %s", component.parent == entt::null ? "false" : "true");
		    ImGui::Text("Children: %s", component.first == entt::null ? "false" : "true");
		    ImGui::Text("Siblings: %s", component.next == entt::null && component.prev == entt::null ? "false" : "true");
	    },
	    true);
}

template <>
inline void draw_component<cmpt::MeshletRenderer>(Entity entity)
{
	draw_component<cmpt::MeshletRenderer>(
	    "MeshletRenderer", entity, [](cmpt::MeshletRenderer &component) {
		    ImGui::Text("Model: ");
		    ImGui::SameLine();
		    ImGui::PushStyleVar(ImGuiStyleVar_ButtonTextAlign, ImVec2(0.f, 0.f));
		    if (ImGui::Button(component.model.c_str(), component.model.empty() ? ImVec2(250.f, 0.f) : ImVec2(0.f, 0.f)))
		    {
			    component.model = "";
			    component.materials.clear();
		    }
		    ImGui::PopStyleVar();
		    if (ImGui::BeginDragDropTarget())
		    {
			    if (const auto *pay_load = ImGui::AcceptDragDropPayload("Model"))
			    {
				    ASSERT(pay_load->DataSize == sizeof(std::string));
				    std::string new_model = *static_cast<std::string *>(pay_load->Data);
				    if (component.model != new_model)
				    {
					    component.model = new_model;
					    component.materials.clear();
					    auto &model = Renderer::instance()->getResourceCache().loadModel(component.model);
					    for (auto &submesh : model.get().submeshes)
					    {
						    component.materials.emplace_back(createScope<material::DisneyPBR>());
						    *static_cast<material::DisneyPBR *>(component.materials.back().get()) = submesh.material;
					    }
				    }
			    }
			    ImGui::EndDragDropTarget();
		    }

		    if (Renderer::instance()->getResourceCache().hasModel(component.model))
		    {
			    auto &model = Renderer::instance()->getResourceCache().loadModel(component.model);

			    uint32_t idx = 0;
			    for (uint32_t i = 0; i < model.get().submeshes.size(); i++)
			    {
				    auto &submesh = model.get().submeshes[i];
				    if (component.materials.size() <= i)
				    {
					    component.materials.emplace_back(createScope<material::DisneyPBR>());
				    }
				    auto &material = component.materials[i];

				    if (ImGui::TreeNode((std::string("Submesh #") + std::to_string(idx++)).c_str()))
				    {
					    // Submesh attributes
					    if (ImGui::TreeNode("Mesh Attributes"))
					    {
						    ImGui::Text("vertices count: %d", submesh.vertices_count);
						    ImGui::Text("vertices offset: %d", submesh.vertices_offset);
						    ImGui::Text("indices count: %d", submesh.indices_count);
						    ImGui::Text("indices offset: %d", submesh.indices_offset);
						    ImGui::Text("meshlets count: %d", submesh.meshlet_count);
						    ImGui::Text("meshlets offset: %d", submesh.meshlet_offset);
						    ImGui::Text("AABB bounding box:");
						    ImGui::BulletText("min (%f, %f, %f)", submesh.bounding_box.min_.x, submesh.bounding_box.min_.y, submesh.bounding_box.min_.z);
						    ImGui::BulletText("max (%f, %f, %f)", submesh.bounding_box.max_.x, submesh.bounding_box.max_.y, submesh.bounding_box.max_.z);
						    ImGui::TreePop();
					    }

					    // Material attributes
					    if (ImGui::TreeNode("Material Attributes"))
					    {
						    // Switch material type
						    if (ImGui::Button(material ? material->type().name() : "Select Material"))
						    {
							    ImGui::OpenPopup("Material Type");
						    }

						    if (ImGui::BeginPopup("Material Type"))
						    {
							    if (material && ImGui::MenuItem("clear"))
							    {
								    material = nullptr;
							    }
							    select_material<material::DisneyPBR>(material);
							    ImGui::EndPopup();
						    }
						    draw_material(material);
						    ImGui::TreePop();
					    }
					    ImGui::TreePop();
				    }
			    }
		    }
	    });
}

template <>
inline void draw_component<cmpt::MeshRenderer>(Entity entity)
{
	draw_component<cmpt::MeshRenderer>("MeshRenderer", entity, [](cmpt::MeshRenderer &component) {
		const char *const MeshNames[] = {"None", "Model", "Sphere", "Plane"};
		int               current     = static_cast<int>(component.type);
		if (ImGui::Combo("Type", &current, MeshNames, 4) && current != static_cast<int>(component.type))
		{
			if (component.type == cmpt::MeshType::Sphere)
			{
				geometry::Sphere sphere({0.f, 0.f, 0.f}, 1.f);
				auto             mesh = std::move(sphere.toTriMesh());
				component.vertices    = std::move(mesh.vertices);
				component.indices     = std::move(mesh.indices);
			}

			component.type = static_cast<cmpt::MeshType>(current);
		}

		if (component.type == cmpt::MeshType::Model)
		{
			if (ImGui::BeginDragDropTarget())
			{
				if (const auto *pay_load = ImGui::AcceptDragDropPayload("Model"))
				{
					ASSERT(pay_load->DataSize == sizeof(std::string));
					std::string new_model = *static_cast<std::string *>(pay_load->Data);
					auto &      model     = Renderer::instance()->getResourceCache().loadModel(new_model);
					if (model.get().submeshes.size() > 1)
					{
						LOG_WARN("Dynamic meshrenderer only support single submesh");
					}
					else
					{
						component.vertices = model.get().mesh.vertices;
						component.indices  = model.get().mesh.indices;
					}
				}
				ImGui::EndDragDropTarget();
			}
		}

		ImGui::Text("vertices count: %d", component.vertices.size());
		ImGui::Text("indices count: %d", component.indices.size());

		if (ImGui::Button(component.material ? component.material->type().name() : "Select Material"))
		{
			ImGui::OpenPopup("Material Type");
		}

		if (ImGui::BeginPopup("Material Type"))
		{
			if (component.material && ImGui::MenuItem("clear"))
			{
				component.material = nullptr;
			}
			select_material<material::DisneyPBR>(component.material);
			ImGui::EndPopup();
		}
		draw_material<material::DisneyPBR>(component.material);
	});
}

template <>
inline void draw_component<cmpt::DirectionalLight>(Entity entity)
{
	draw_component<cmpt::DirectionalLight>("Directional Light", entity, [](cmpt::DirectionalLight &component) {
		ImGui::DragFloat("Intensity", &component.intensity, 0.01f, 0.f, std::numeric_limits<float>::max(), "%.3f");
		ImGui::ColorEdit3("Color", glm::value_ptr(component.color));
		ImGui::DragFloat3("Direction", glm::value_ptr(component.direction), 0.1f, 0.0f, 0.0f, "%.3f");
	});
}

template <>
inline void draw_component<cmpt::SpotLight>(Entity entity)
{
	draw_component<cmpt::SpotLight>("Spot Light", entity, [](cmpt::SpotLight &component) {
		ImGui::DragFloat("Intensity", &component.intensity, 0.01f, 0.f, std::numeric_limits<float>::max(), "%.3f");
		ImGui::ColorEdit3("Color", glm::value_ptr(component.color));
		ImGui::DragFloat3("Direction", glm::value_ptr(component.direction), 0.1f, 0.0f, 0.0f, "%.3f");
		ImGui::DragFloat("Cut off", &component.cut_off, 0.0001f, 0.f, std::numeric_limits<float>::max(), "%.5f");
		ImGui::DragFloat("Outer cut off", &component.outer_cut_off, 0.0001f, 0.f, std::numeric_limits<float>::max(), "%.5f");
	});
}

template <>
inline void draw_component<cmpt::PointLight>(Entity entity)
{
	draw_component<cmpt::PointLight>("Point Light", entity, [](cmpt::PointLight &component) {
		ImGui::DragFloat("Intensity", &component.intensity, 0.01f, 0.f, std::numeric_limits<float>::max(), "%.3f");
		ImGui::ColorEdit3("Color", glm::value_ptr(component.color));
		ImGui::DragFloat("Constant", &component.constant, 0.0001f, 0.f, std::numeric_limits<float>::max(), "%.5f");
		ImGui::DragFloat("Linear", &component.linear, 0.0001f, 0.f, std::numeric_limits<float>::max(), "%.5f");
		ImGui::DragFloat("Quadratic", &component.quadratic, 0.0001f, 0.f, std::numeric_limits<float>::max(), "%.5f");
	});
}

template <>
inline void draw_component<cmpt::PerspectiveCamera>(Entity entity)
{
	draw_component<cmpt::PerspectiveCamera>("Perspective Camera", entity, [](cmpt::PerspectiveCamera &component) {

		ImGui::DragFloat("Aspect", &component.aspect, 0.01f, 0.f, std::numeric_limits<float>::max(), "%.3f");
		ImGui::DragFloat("Fov", &component.fov, 0.01f, 0.f, 90.f, "%.3f");
		ImGui::DragFloat("Near Plane", &component.near_plane, 0.01f, 0.f, std::numeric_limits<float>::max(), "%.3f");
		ImGui::DragFloat("Far Plane", &component.far_plane, 0.01f, 0.f, std::numeric_limits<float>::max(), "%.3f");
	});
}

template <>
inline void draw_component<cmpt::OrthographicCamera>(Entity entity)
{
	draw_component<cmpt::OrthographicCamera>("Orthographic Camera", entity, [](cmpt::OrthographicCamera &component) {
		ImGui::DragFloat("Left", &component.left, 0.01f, -std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), "%.3f");
		ImGui::DragFloat("Right", &component.right, 0.01f, -std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), "%.3f");
		ImGui::DragFloat("Bottom", &component.bottom, 0.01f, -std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), "%.3f");
		ImGui::DragFloat("Top", &component.top, 0.01f, -std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), "%.3f");
		ImGui::DragFloat("Near Plane", &component.near_plane, 0.01f, 0.f, std::numeric_limits<float>::max(), "%.3f");
		ImGui::DragFloat("Far Plane", &component.far_plane, 0.01f, 0.f, std::numeric_limits<float>::max(), "%.3f");
	});
}

Inspector::Inspector()
{
	m_name = "Inspector";
}

void Inspector::draw(float delta_time)
{
	ImGui::Begin("Inspector", &active);

	auto entity = Editor::instance()->getSelect();

	if (!entity.valid())
	{
		ImGui::End();
		return;
	}

	// Editable tag
	draw_component<cmpt::Tag>(entity);

	ImGui::SameLine();
	ImGui::PushItemWidth(-1);

	// Add components popup
	if (ImGui::Button("Add Component"))
	{
		ImGui::OpenPopup("AddComponent");
	}

	if (ImGui::BeginPopup("AddComponent"))
	{
		add_component<cmpt::Light, cmpt::DirectionalLight, cmpt::PointLight, cmpt::SpotLight>();
		add_component<cmpt::Renderable, cmpt::MeshletRenderer, cmpt::MeshRenderer>();
		add_component<cmpt::Camera, cmpt::PerspectiveCamera, cmpt::OrthographicCamera>();
		ImGui::EndPopup();
	}
	ImGui::PopItemWidth();

	draw_component<cmpt::Transform, cmpt::Hierarchy, cmpt::MeshletRenderer, cmpt::MeshRenderer, 
		cmpt::DirectionalLight, cmpt::PointLight, cmpt::SpotLight, cmpt::PerspectiveCamera, cmpt::OrthographicCamera>(entity);

	ImGui::End();
}
}        // namespace Ilum::panel