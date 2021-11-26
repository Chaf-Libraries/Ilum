#include "Inspector.hpp"

#include "Editor/Editor.hpp"

#include "Scene/Component/DirectionalLight.hpp"
#include "Scene/Component/Hierarchy.hpp"
#include "Scene/Component/Light.hpp"
#include "Scene/Component/MeshRenderer.hpp"
#include "Scene/Component/PointLight.hpp"
#include "Scene/Component/SpotLight.hpp"
#include "Scene/Component/Tag.hpp"
#include "Scene/Component/Transform.hpp"

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
	update = update | ImGui::DragFloat("##X", &values.x, 0.1f, 0.0f, 0.0f, "%.3f");
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
	update = update | ImGui::DragFloat("##Y", &values.y, 0.1f, 0.0f, 0.0f, "%.3f");
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
	update = update | ImGui::DragFloat("##Z", &values.z, 0.1f, 0.0f, 0.0f, "%.3f");
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
	if (ImGui::ImageButton(Renderer::instance()->getResourceCache().hasImage(texture) ?
                               ImGuiContext::textureID(Renderer::instance()->getResourceCache().loadImage(texture), Renderer::instance()->getSampler(Renderer::SamplerType::Trilinear_Clamp)) :
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
	ImGui::ColorEdit4("Base Color", glm::value_ptr(material.base_color));
	ImGui::ColorEdit3("Emissive Color", glm::value_ptr(material.emissive_color));
	ImGui::DragFloat("Metallic Factor", &material.metallic_factor, 0.01f, 0.f, 1.f, "%.3f");
	ImGui::DragFloat("Emissive Intensity", &material.emissive_intensity, 0.01f, 0.f, std::numeric_limits<float>::max(), "%.3f");
	ImGui::DragFloat("Roughness Factor", &material.roughness_factor, 0.01f, 0.f, 1.f, "%.3f");
	ImGui::DragFloat("Height Factor", &material.displacement_height, 0.01f, 0.f, std::numeric_limits<float>::max(), "%.3f");

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
inline void add_component()
{
	if (!Editor::instance()->getSelect().hasComponent<T>() && ImGui::MenuItem(typeid(T).name()))
	{
		Editor::instance()->getSelect().addComponent<T>();
		ImGui::CloseCurrentPopup();
	}
}

template <typename T1, typename T2, typename... Tn>
inline void add_component()
{
	add_component<T1>();
	add_component<T2, Tn...>();
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
		ImGui::Checkbox("Active", &entity.getComponent<cmpt::Tag>().active);
	}
}

template <>
inline void draw_component<cmpt::Transform>(Entity entity)
{
	draw_component<cmpt::Transform>(
	    "Transform", entity, [](auto &component) {
		    component.update = draw_vec3_control("Translation", component.translation, 0.f);
		    component.update = component.update | draw_vec3_control("Rotation", component.rotation, 0.f);
		    component.update = component.update | draw_vec3_control("Scale", component.scale, 1.f);
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
inline void draw_component<cmpt::MeshRenderer>(Entity entity)
{
	draw_component<cmpt::MeshRenderer>(
	    "MeshRenderer", entity, [](cmpt::MeshRenderer &component) {
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
inline void draw_component<cmpt::Light>(Entity entity)
{
	draw_component<cmpt::Light>(
	    "Light", entity, [](cmpt::Light &component) {
		    const char *const LightNames[] = {"None", "Directional", "Point", "Spot"};
		    int               current      = static_cast<int>(component.type);
		    ImGui::Combo("Type", &current, LightNames, 4);

		    if (component.type != cmpt::LightType::None && !component.impl)
		    {
			    return;
		    }

		    if (component.type == cmpt::LightType::Directional)
		    {
			    auto light = static_cast<cmpt::DirectionalLight *>(component.impl.get());
			    ImGui::DragFloat("Intensity", &light->data.intensity, 0.01f, 0.f, std::numeric_limits<float>::max(), "%.3f");
			    ImGui::ColorEdit3("Color", glm::value_ptr(light->data.color));
			    ImGui::DragFloat3("Direction", glm::value_ptr(light->data.direction), 0.1f, 0.0f, 0.0f, "%.3f");
		    }
		    else if (component.type == cmpt::LightType::Spot)
		    {
			    auto light = static_cast<cmpt::SpotLight *>(component.impl.get());
			    ImGui::DragFloat("Intensity", &light->data.intensity, 0.01f, 0.f, std::numeric_limits<float>::max(), "%.3f");
			    ImGui::ColorEdit3("Color", glm::value_ptr(light->data.color));
			    ImGui::DragFloat3("Direction", glm::value_ptr(light->data.direction), 0.1f, 0.0f, 0.0f, "%.3f");
			    ImGui::DragFloat("Cut off", &light->data.cut_off, 0.0001f, 0.f, std::numeric_limits<float>::max(), "%.5f");
			    ImGui::DragFloat("Outer cut off", &light->data.outer_cut_off, 0.0001f, 0.f, std::numeric_limits<float>::max(), "%.5f");
		    }
		    else if (component.type == cmpt::LightType::Point)
		    {
			    auto light = static_cast<cmpt::PointLight *>(component.impl.get());
			    ImGui::DragFloat("Intensity", &light->data.intensity, 0.01f, 0.f, std::numeric_limits<float>::max(), "%.3f");
			    ImGui::ColorEdit3("Color", glm::value_ptr(light->data.color));
			    ImGui::DragFloat("Constant", &light->data.constant, 0.0001f, 0.f, std::numeric_limits<float>::max(), "%.5f");
			    ImGui::DragFloat("Linear", &light->data.linear, 0.0001f, 0.f, std::numeric_limits<float>::max(), "%.5f");
			    ImGui::DragFloat("Quadratic", &light->data.quadratic, 0.0001f, 0.f, std::numeric_limits<float>::max(), "%.5f");
		    }
		    else if (component.type == cmpt::LightType::Area)
		    {
			    ImGui::Text("Area");
		    }

		    component.type = static_cast<cmpt::LightType>(current);
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
		add_component<cmpt::MeshRenderer, cmpt::Light>();
		ImGui::EndPopup();
	}
	ImGui::PopItemWidth();

	draw_component<cmpt::Transform, cmpt::Hierarchy, cmpt::MeshRenderer, cmpt::Light>(entity);

	ImGui::End();
}
}        // namespace Ilum::panel