#include "Inspector.hpp"

#include "Editor/Editor.hpp"

#include "Scene/Component/Camera.hpp"
#include "Scene/Component/Hierarchy.hpp"
#include "Scene/Component/Light.hpp"
#include "Scene/Component/Renderable.hpp"
#include "Scene/Component/Tag.hpp"
#include "Scene/Component/Transform.hpp"

#include "Geometry/Shape/Sphere.hpp"

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

template <typename T, typename Callback>
inline bool draw_component(const std::string &name, Entity entity, Callback callback, bool static_mode = false)
{
	const ImGuiTreeNodeFlags tree_node_flags = ImGuiTreeNodeFlags_DefaultOpen | ImGuiTreeNodeFlags_Framed | ImGuiTreeNodeFlags_SpanAvailWidth | ImGuiTreeNodeFlags_AllowItemOverlap | ImGuiTreeNodeFlags_FramePadding;
	if (entity.hasComponent<T>())
	{
		auto  &component                = entity.getComponent<T>();
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
			update = callback(component);
			ImGui::TreePop();
		}

		if (remove_component)
		{
			entity.removeComponent<T>();
			update = true;
		}

		return update;
	}

	return false;
}

template <typename T>
inline void draw_material(T &material)
{
	ASSERT(false);
}

bool draw_texture(std::string &texture, const std::string &name)
{
	bool update = false;
	ImGui::PushID(name.c_str());
	if (ImGui::ImageButton(Renderer::instance()->getResourceCache().hasImage(FileSystem::getRelativePath(texture)) ?
	                           ImGuiContext::textureID(Renderer::instance()->getResourceCache().loadImage(FileSystem::getRelativePath(texture)), Renderer::instance()->getSampler(Renderer::SamplerType::Trilinear_Clamp)) :
                               ImGuiContext::textureID(Renderer::instance()->getDefaultTexture(), Renderer::instance()->getSampler(Renderer::SamplerType::Trilinear_Clamp)),
	                       ImVec2{100.f, 100.f}))
	{
		texture = "";
		update  = true;
	}
	ImGui::PopID();

	if (ImGui::BeginDragDropTarget())
	{
		if (const auto *pay_load = ImGui::AcceptDragDropPayload("Texture2D"))
		{
			ASSERT(pay_load->DataSize == sizeof(std::string));
			if (texture != *static_cast<std::string *>(pay_load->Data))
			{
				texture = *static_cast<std::string *>(pay_load->Data);
				update  = true;
			}
		}
		ImGui::EndDragDropTarget();
	}

	return update;
}

template <BxDFType _Ty>
inline void draw_material(Material &material)
{
}

template <>
inline void draw_material<BxDFType::Disney>(Material &material)
{
	Material::update = ImGui::ColorEdit4("Base Color", glm::value_ptr(material.base_color)) || Material::update;
	Material::update = ImGui::ColorEdit3("Scatter Distance", glm::value_ptr(material.data)) || Material::update;
	Material::update = ImGui::ColorEdit3("Emissive", glm::value_ptr(material.emissive_color)) || Material::update;
	Material::update = ImGui::DragFloat("Emissive Intensity", &material.emissive_intensity, 0.01f, 0.f, std::numeric_limits<float>::max(), "%.3f") || Material::update;
	Material::update = ImGui::DragFloat("Metallic", &material.metallic, 0.001f, 0.f, 1.f, "%.3f") || Material::update;
	Material::update = ImGui::DragFloat("Roughness", &material.roughness, 0.001f, 0.f, 1.f, "%.3f") || Material::update;
	Material::update = ImGui::DragFloat("Subsurface", &material.subsurface, 0.001f, 0.f, 1.f, "%.3f") || Material::update;
	Material::update = ImGui::DragFloat("Specular", &material.specular, 0.001f, 0.f, 1.f, "%.3f") || Material::update;
	Material::update = ImGui::DragFloat("Specular Tint", &material.specular_tint, 0.001f, 0.f, 1.f, "%.3f") || Material::update;
	Material::update = ImGui::DragFloat("Anisotropic", &material.anisotropic, 0.001f, 0.f, 1.f, "%.3f") || Material::update;
	Material::update = ImGui::DragFloat("Sheen", &material.sheen, 0.001f, 0.f, 1.f, "%.3f") || Material::update;
	Material::update = ImGui::DragFloat("Sheen Tint", &material.sheen_tint, 0.001f, 0.f, 1.f, "%.3f") || Material::update;
	Material::update = ImGui::DragFloat("Clearcoat", &material.clearcoat, 0.001f, 0.f, 1.f, "%.3f") || Material::update;
	Material::update = ImGui::DragFloat("Clearcoat Gloss", &material.clearcoat_gloss, 0.001f, 0.f, 1.f, "%.3f") || Material::update;
	Material::update = ImGui::DragFloat("Specular Transmission", &material.specular_transmission, 0.001f, 0.f, 1.f, "%.3f") || Material::update;
	Material::update = ImGui::DragFloat("Diffuse Transmission", &material.diffuse_transmission, 0.001f, 0.f, 1.f, "%.3f") || Material::update;
	Material::update = ImGui::DragFloat("Flatness", &material.flatness, 0.001f, 0.f, 1.f, "%.3f") || Material::update;
	Material::update = ImGui::DragFloat("Refraction", &material.refraction, 0.001f, 0.f, std::numeric_limits<float>::max(), "%.3f") || Material::update;
	Material::update = ImGui::DragFloat("Displacement", &material.displacement, 0.001f, 0.f, std::numeric_limits<float>::max(), "%.3f") || Material::update;

	bool thin        = material.thin > 0.0;
	Material::update = ImGui::Checkbox("Thin", &thin) || Material::update;
	material.thin    = thin ? 1.f : 0.f;

	ImGui::Text("Albedo Map");
	Material::update = draw_texture(material.textures[TextureType::BaseColor], "Albedo Map") || Material::update;

	ImGui::Text("Normal Map");
	Material::update = draw_texture(material.textures[TextureType::Normal], "Normal Map") || Material::update;

	ImGui::Text("Metallic Map");
	Material::update = draw_texture(material.textures[TextureType::Metallic], "Metallic Map") || Material::update;

	ImGui::Text("Roughness Map");
	Material::update = draw_texture(material.textures[TextureType::Roughness], "Roughness Map") || Material::update;

	ImGui::Text("Emissive Map");
	Material::update = draw_texture(material.textures[TextureType::Emissive], "Emissive Map") || Material::update;

	ImGui::Text("AO Map");
	Material::update = draw_texture(material.textures[TextureType::AmbientOcclusion], "AO Map") || Material::update;

	ImGui::Text("Displacement Map");
	Material::update = draw_texture(material.textures[TextureType::Displacement], "Displacement Map") || Material::update;
}

template <>
inline void draw_material<BxDFType::CookTorrance>(Material &material)
{
	Material::update = ImGui::ColorEdit4("Base Color", glm::value_ptr(material.base_color)) || Material::update;
	Material::update = ImGui::ColorEdit3("Emissive", glm::value_ptr(material.emissive_color)) || Material::update;
	Material::update = ImGui::DragFloat("Emissive Intensity", &material.emissive_intensity, 0.01f, 0.f, std::numeric_limits<float>::max(), "%.3f") || Material::update;
	Material::update = ImGui::DragFloat("Metallic", &material.metallic, 0.001f, 0.f, 1.f, "%.3f") || Material::update;
	Material::update = ImGui::DragFloat("Roughness", &material.roughness, 0.001f, 0.f, 1.f, "%.3f") || Material::update;
	Material::update = ImGui::DragFloat("Displacement", &material.displacement, 0.001f, 0.f, std::numeric_limits<float>::max(), "%.3f") || Material::update;

	ImGui::Text("Albedo Map");
	Material::update = draw_texture(material.textures[TextureType::BaseColor], "Albedo Map") || Material::update;

	ImGui::Text("Normal Map");
	Material::update = draw_texture(material.textures[TextureType::Normal], "Normal Map") || Material::update;

	ImGui::Text("Metallic Map");
	Material::update = draw_texture(material.textures[TextureType::Metallic], "Metallic Map") || Material::update;

	ImGui::Text("Roughness Map");
	Material::update = draw_texture(material.textures[TextureType::Roughness], "Roughness Map") || Material::update;

	ImGui::Text("Emissive Map");
	Material::update = draw_texture(material.textures[TextureType::Emissive], "Emissive Map") || Material::update;

	ImGui::Text("AO Map");
	Material::update = draw_texture(material.textures[TextureType::AmbientOcclusion], "AO Map") || Material::update;

	ImGui::Text("Displacement Map");
	Material::update = draw_texture(material.textures[TextureType::Displacement], "Displacement Map") || Material::update;
}

template <>
inline void draw_material<BxDFType::Matte>(Material &material)
{
	Material::update = ImGui::ColorEdit3("Base Color", glm::value_ptr(material.base_color)) || Material::update;
	Material::update = ImGui::DragFloat("Roughness", &material.roughness, 0.001f, 0.f, 90.f, "%.3f") || Material::update;
	Material::update = draw_texture(material.textures[TextureType::BaseColor], "Albedo Map") || Material::update;

	ImGui::Text("Albedo Map");
	Material::update = draw_texture(material.textures[TextureType::BaseColor], "Albedo Map") || Material::update;

	ImGui::Text("Normal Map");
	Material::update = draw_texture(material.textures[TextureType::Normal], "Normal Map") || Material::update;

	ImGui::Text("Roughness Map");
	Material::update = draw_texture(material.textures[TextureType::Roughness], "Roughness Map") || Material::update;
}

template <>
inline void draw_material<BxDFType::Plastic>(Material &material)
{
	Material::update = ImGui::ColorEdit3("Base Color", glm::value_ptr(material.base_color)) || Material::update;
	Material::update = ImGui::ColorEdit3("Specular", glm::value_ptr(material.data)) || Material::update;
	Material::update = ImGui::DragFloat("Roughness", &material.roughness, 0.001f, 0.f, 1.f, "%.3f") || Material::update;

	ImGui::Text("Albedo Map");
	Material::update = draw_texture(material.textures[TextureType::BaseColor], "Albedo Map") || Material::update;

	ImGui::Text("Normal Map");
	Material::update = draw_texture(material.textures[TextureType::Normal], "Normal Map") || Material::update;

	ImGui::Text("Roughness Map");
	Material::update = draw_texture(material.textures[TextureType::Roughness], "Roughness Map") || Material::update;
}

template <>
inline void draw_material<BxDFType::Metal>(Material &material)
{
	Material::update = ImGui::ColorEdit3("Base Color", glm::value_ptr(material.base_color)) || Material::update;
	Material::update = ImGui::DragFloat("Roughness", &material.roughness, 0.001f, 0.f, 1.f, "%.3f") || Material::update;
	Material::update = ImGui::DragFloat("Anisotropic", &material.anisotropic, 0.001f, -1.f, 1.f, "%.3f") || Material::update;

	ImGui::Text("Albedo Map");
	Material::update = draw_texture(material.textures[TextureType::BaseColor], "Albedo Map") || Material::update;

	ImGui::Text("Normal Map");
	Material::update = draw_texture(material.textures[TextureType::Normal], "Normal Map") || Material::update;

	ImGui::Text("Roughness Map");
	Material::update = draw_texture(material.textures[TextureType::Roughness], "Roughness Map") || Material::update;
}

template <>
inline void draw_material<BxDFType::Mirror>(Material &material)
{
	Material::update = ImGui::ColorEdit3("Base Color", glm::value_ptr(material.base_color)) || Material::update;

	ImGui::Text("Albedo Map");
	Material::update = draw_texture(material.textures[TextureType::BaseColor], "Albedo Map") || Material::update;

	ImGui::Text("Normal Map");
	Material::update = draw_texture(material.textures[TextureType::Normal], "Normal Map") || Material::update;
}

template <>
inline void draw_material<BxDFType::Substrate>(Material &material)
{
	Material::update = ImGui::ColorEdit3("Diffuse Color", glm::value_ptr(material.base_color)) || Material::update;
	Material::update = ImGui::ColorEdit3("Glossy Color", glm::value_ptr(material.data)) || Material::update;
	Material::update = ImGui::DragFloat("Roughness", &material.roughness, 0.001f, 0.f, 1.f, "%.3f") || Material::update;
	Material::update = ImGui::DragFloat("Anisotropic", &material.anisotropic, 0.001f, -1.f, 1.f, "%.3f") || Material::update;

	ImGui::Text("Albedo Map");
	Material::update = draw_texture(material.textures[TextureType::BaseColor], "Albedo Map") || Material::update;

	ImGui::Text("Normal Map");
	Material::update = draw_texture(material.textures[TextureType::Normal], "Normal Map") || Material::update;
}

template <>
inline void draw_material<BxDFType::Glass>(Material &material)
{
	Material::update = ImGui::ColorEdit3("Reflection Color", glm::value_ptr(material.base_color)) || Material::update;
	Material::update = ImGui::ColorEdit3("Transmission Color", glm::value_ptr(material.data)) || Material::update;
	Material::update = ImGui::DragFloat("Refraction", &material.refraction, 0.001f, std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), "%.3f") || Material::update;
	Material::update = ImGui::DragFloat("Roughness", &material.roughness, 0.001f, 0.f, 1.f, "%.3f") || Material::update;
	Material::update = ImGui::DragFloat("Anisotropic", &material.anisotropic, 0.001f, -1.f, 1.f, "%.3f") || Material::update;

	ImGui::Text("Albedo Map");
	Material::update = draw_texture(material.textures[TextureType::BaseColor], "Albedo Map") || Material::update;

	ImGui::Text("Normal Map");
	Material::update = draw_texture(material.textures[TextureType::Normal], "Normal Map") || Material::update;
}

inline void draw_material(Material &material)
{
	const char *const BxDF_types[] = {
	    "Matte",
	    "Plastic",
	    "Metal",
	    "Mirror",
	    "Substrate",
	    "Glass",
	    "Disney",
	    "CookTorrance",
	};
	Material::update = ImGui::Combo("BxDF", reinterpret_cast<int *>(&material.type), BxDF_types, 8) || Material::update;

	switch (material.type)
	{
		case BxDFType::Disney:
			draw_material<BxDFType::Disney>(material);
			break;
		case BxDFType::CookTorrance:
			draw_material<BxDFType::CookTorrance>(material);
			break;
		case BxDFType::Matte:
			draw_material<BxDFType::Matte>(material);
			break;
		case BxDFType::Plastic:
			draw_material<BxDFType::Plastic>(material);
			break;
		case BxDFType::Metal:
			draw_material<BxDFType::Metal>(material);
			break;
		case BxDFType::Mirror:
			draw_material<BxDFType::Mirror>(material);
			break;
		case BxDFType::Substrate:
			draw_material<BxDFType::Substrate>(material);
			break;
		case BxDFType::Glass:
			draw_material<BxDFType::Glass>(material);
			break;
		default:
			break;
	}
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
inline bool draw_component(Entity entity)
{
	return false;
}

template <typename T1, typename T2, typename... Tn>
inline bool draw_component(Entity entity)
{
	bool update = false;
	update      = draw_component<T1>(entity) || update;
	update      = draw_component<T2, Tn...>(entity) || update;
	return update;
}

template <>
inline bool draw_component<cmpt::Tag>(Entity entity)
{
	bool update = false;
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
			tag    = std::string(buffer);
			update = true;
		}
		ImGui::PopItemWidth();
	}
	return update;
}

template <>
inline bool draw_component<cmpt::Transform>(Entity entity)
{
	return draw_component<cmpt::Transform>(
	    "Transform", entity, [](cmpt::Transform &component) {
		    component.update = draw_vec3_control("Translation", component.translation, 0.f) || component.update;
		    component.update = draw_vec3_control("Rotation", component.rotation, 0.f) || component.update;
		    component.update = draw_vec3_control("Scale", component.scale, 1.f) || component.update;
		    return component.update;
	    },
	    true);
}

template <>
inline bool draw_component<cmpt::Hierarchy>(Entity entity)
{
	return draw_component<cmpt::Hierarchy>(
	    "Hierarchy", entity, [](auto &component) {
		    ImGui::Text("Parent: %s", component.parent == entt::null ? "false" : "true");
		    ImGui::Text("Children: %s", component.first == entt::null ? "false" : "true");
		    ImGui::Text("Siblings: %s", component.next == entt::null && component.prev == entt::null ? "false" : "true");
		    return false;
	    },
	    true);
}

template <>
inline bool draw_component<cmpt::StaticMeshRenderer>(Entity entity)
{
	return draw_component<cmpt::StaticMeshRenderer>(
	    "StaticMeshRenderer", entity, [](cmpt::StaticMeshRenderer &component) {
		    ImGui::Text("Model: ");
		    ImGui::SameLine();
		    ImGui::PushStyleVar(ImGuiStyleVar_ButtonTextAlign, ImVec2(0.f, 0.f));
		    if (ImGui::Button(component.model.c_str(), component.model.empty() ? ImVec2(250.f, 0.f) : ImVec2(0.f, 0.f)))
		    {
			    component.model = "";
			    component.materials.clear();
			    cmpt::StaticMeshRenderer::update = true;
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
					    cmpt::StaticMeshRenderer::update = true;
					    component.model                  = new_model;
					    component.materials.clear();
					    auto &model = Renderer::instance()->getResourceCache().loadModel(component.model);
					    for (auto &submesh : model.get().submeshes)
					    {
						    component.materials.emplace_back(submesh.material);
					    }
					    Material::update = true;
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
					    component.materials.emplace_back(Material());
				    }
				    auto &material = component.materials[i];

				    ImGui::PushID((std::to_string(idx) + "Submesh").c_str());
				    if (ImGui::TreeNode(submesh.name.c_str()))
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
						    draw_material(material);
						    ImGui::TreePop();
					    }
					    ImGui::TreePop();
				    }
				    ImGui::PopID();
			    }
		    }
		    return cmpt::StaticMeshRenderer::update || Material::update;
	    });
}

template <>
inline bool draw_component<cmpt::DynamicMeshRenderer>(Entity entity)
{
	return draw_component<cmpt::DynamicMeshRenderer>("DynamicMeshRenderer", entity, [](cmpt::DynamicMeshRenderer &component) {
		const char *const mesh_names[] = {"None", "Model", "Sphere", "Plane"};
		int               current      = static_cast<int>(component.type);
		if (ImGui::Combo("Type", &current, mesh_names, 4) && current != static_cast<int>(component.type))
		{
			if (component.type == cmpt::MeshType::Sphere)
			{
				geometry::Sphere sphere({0.f, 0.f, 0.f}, 1.f);
				auto [vertices, indices] = sphere.toMesh();
				component.vertices       = std::move(vertices);
				component.indices        = std::move(indices);
			}

			component.type        = static_cast<cmpt::MeshType>(current);
			component.need_update = true;
		}

		if (component.type == cmpt::MeshType::Model)
		{
			if (ImGui::BeginDragDropTarget())
			{
				if (const auto *pay_load = ImGui::AcceptDragDropPayload("Model"))
				{
					ASSERT(pay_load->DataSize == sizeof(std::string));
					std::string new_model = *static_cast<std::string *>(pay_load->Data);
					auto       &model     = Renderer::instance()->getResourceCache().loadModel(new_model);
					if (model.get().submeshes.size() > 1)
					{
						LOG_WARN("Dynamic meshrenderer only support single submesh");
					}
					else
					{
						component.vertices    = model.get().vertices;
						component.indices     = model.get().indices;
						component.material    = model.get().submeshes[0].material;
						component.bbox        = model.get().bounding_box;
						component.need_update = true;
						Material::update      = true;
					}
				}
				ImGui::EndDragDropTarget();
			}
		}

		ImGui::Text("vertices count: %d", component.vertices.size());
		ImGui::Text("indices count: %d", component.indices.size());

		draw_material(component.material);

		return component.need_update || Material::update;
	});
}

template <>
inline bool draw_component<cmpt::CurveRenderer>(Entity entity)
{
	return draw_component<cmpt::CurveRenderer>("CurveRenderer", entity, [](cmpt::CurveRenderer &component) {
		const char *const curve_names[] = {"None", "Bezier Curve", "B Spline", "Cubic Spline", "Rational Bezier", "Rational B Spline"};
		int               current       = static_cast<int>(component.type);
		if (ImGui::Combo("Type", &current, curve_names, 6) && current != static_cast<int>(component.type))
		{
			component.type        = static_cast<cmpt::CurveType>(current);
			component.need_update = true;
		}

		ImGui::ColorEdit4("Color", glm::value_ptr(component.base_color));
		ImGui::DragFloat("Width", &component.line_width, 0.01f, 0.f, std::numeric_limits<float>::max(), "%.2f");

		int sample = static_cast<int>(component.sample);
		if (ImGui::DragInt("Sample", &sample, 1.f, 0, std::numeric_limits<int>::max()))
		{
			component.sample      = static_cast<uint32_t>(sample);
			component.need_update = true;
		}

		if (current == 2 || current == 5)
		{
			int order = static_cast<int>(component.order);
			if (ImGui::DragInt("Order", &order, 1.f, 1, std::numeric_limits<int>::max()))
			{
				component.order       = static_cast<uint32_t>(order);
				component.need_update = true;
			}
		}

		uint32_t point_idx = 0;
		auto     iter      = component.control_points.begin();
		ImGui::Text("Control Points:");
		while (iter != component.control_points.end())
		{
			draw_vec3_control(std::to_string(point_idx++).c_str(), *iter, 0.f, 30.f);
			ImGui::SameLine();
			ImGui::PushID(point_idx);
			if (ImGui::Button("-"))
			{
				iter                  = component.control_points.erase(iter);
				component.need_update = true;
			}
			else
			{
				iter++;
			}
			if (current == 4 || current == 5)
			{
				if (ImGui::DragFloat("weight", &component.weights[point_idx - 1], 0.01f, -std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), "%.2f"))
				{
					component.need_update = true;
				}
			}
			ImGui::SameLine();
			bool is_select = point_idx == (component.select_point + 1);
			if (ImGui::Checkbox("", &is_select) && is_select)
			{
				component.select_point = point_idx - 1;
			}

			ImGui::PopID();
		}
		if (ImGui::Button("Add Control Points"))
		{
			component.control_points.push_back(glm::vec3(0.f));
			component.weights.push_back(1.f);
			component.need_update = true;
		}

		return component.need_update;
	});
}

template <>
inline bool draw_component<cmpt::SurfaceRenderer>(Entity entity)
{
	return draw_component<cmpt::SurfaceRenderer>("SurfaceRenderer", entity, [](cmpt::SurfaceRenderer &component) {
		const char *const surface_names[] = {
		    "None",
		    "Bezier Surface",
		    "B Spline Surface",
		    "Rational Bezier Spline Surface",
		    "Rational B Spline Surface",
		};
		int current = static_cast<int>(component.type);
		if (ImGui::Combo("Type", &current, surface_names, 5) && current != static_cast<int>(component.type))
		{
			component.type        = static_cast<cmpt::SurfaceType>(current);
			component.need_update = true;
		}

		ImGui::PushItemWidth(ImGui::GetContentRegionAvailWidth() / 4.f);
		int sample = static_cast<int>(component.sample_x);
		if (ImGui::DragInt("SampleX", &sample, 1.f, 0, std::numeric_limits<int>::max()))
		{
			component.sample_x    = static_cast<uint32_t>(sample);
			component.need_update = true;
		}
		ImGui::SameLine();
		sample = static_cast<int>(component.sample_y);
		if (ImGui::DragInt("SampleY", &sample, 1.f, 0, std::numeric_limits<int>::max()))
		{
			component.sample_y    = static_cast<uint32_t>(sample);
			component.need_update = true;
		}

		int  nu                    = static_cast<int>(component.control_points.size());
		int  nv                    = component.control_points.empty() ? 0 : static_cast<int>(component.control_points[0].size());
		bool resize_control_points = false;

		ImGui::Text("Control Points: ");
		if (ImGui::DragInt("NU", &nu, 0.1f, 1, std::numeric_limits<int>::max()))
		{
			resize_control_points = true;
		}
		ImGui::SameLine();
		if (ImGui::DragInt("NV", &nv, 0.1f, 1, std::numeric_limits<int>::max()))
		{
			resize_control_points = true;
		}

		ImGui::PopItemWidth();

		// Reset control points after resize
		if (resize_control_points)
		{
			bool need_extend = false;
			if (component.control_points.size() != nu)
			{
				if (component.control_points.size() <= nu)
				{
					need_extend = true;
				}
				component.control_points.resize(nu);
				component.weights.resize(nu);
				component.need_update = true;
			}

			if (!component.control_points.empty() && (component.control_points[0].size() != nv || need_extend))
			{
				for (auto &points : component.control_points)
				{
					if (points.size() != nv)
					{
						points.resize(nv);
					}
				}
				for (auto &weight : component.weights)
				{
					if (weight.size() != nv)
					{
						weight.resize(nv);
					}
				}
				component.need_update = true;
			}

			for (int x = 0; x < nu; x++)
			{
				for (int y = 0; y < nv; y++)
				{
					component.control_points[x][y] = glm::vec3(static_cast<float>(x) - static_cast<float>(nu) / 2.f, 0.f, static_cast<float>(y) - static_cast<float>(nv) / 2.f);
					component.weights[x][y]        = 1.f;
				}
			}
		}

		for (uint32_t i = 0; i < component.control_points.size(); i++)
		{
			if (ImGui::TreeNode((std::string("Row ") + std::to_string(i)).c_str()))
			{
				for (uint32_t j = 0; j < component.control_points[i].size(); j++)
				{
					ImGui::PushID((std::to_string(i) + std::to_string(j)).c_str());
					draw_vec3_control(std::to_string(i) + std::to_string(j), component.control_points[i][j], 0.f, 30.f);
					if (current == 3 || current == 4)
					{
						if (ImGui::DragFloat("weight", &component.weights[i][j], 0.01f, -std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), "%.2f"))
						{
							component.need_update = true;
						}
					}
					ImGui::SameLine();
					bool is_select = (i == component.select_point[0] && j == component.select_point[1]);
					if (ImGui::Checkbox("", &is_select) && is_select)
					{
						component.select_point[0] = i;
						component.select_point[1] = j;
					}
					ImGui::PopID();
				}
				ImGui::TreePop();
			}
		}

		if (current == 2 || current == 4)
		{
			int order = static_cast<int>(component.order);
			if (ImGui::DragInt("Order", &order, 1.f, 1, std::numeric_limits<int>::max()))
			{
				component.order       = static_cast<uint32_t>(order);
				component.need_update = true;
			}
		}

		return component.need_update;
	});
}

template <>
inline bool draw_component<cmpt::DirectionalLight>(Entity entity)
{
	return draw_component<cmpt::DirectionalLight>("Directional Light", entity, [](cmpt::DirectionalLight &component) {
		const char *const sample_method[] = {"Uniform", "Possion"};
		const char *const shadow_mode[]   = {"No Shadow", "Hard Shadow", "PCF", "PCSS"};

		bool update = false;

		update = ImGui::DragFloat("Intensity", &component.intensity, 0.01f, 0.f, std::numeric_limits<float>::max(), "%.3f") || update;
		update = ImGui::ColorEdit3("Color", glm::value_ptr(component.color)) || update;
		update = ImGui::DragFloat("Light size", &component.light_size, 0.01f, 0.f, std::numeric_limits<float>::max(), "%.2f") || update;
		ImGui::Text("Shadow Setting");
		update = ImGui::Combo("Shadow Mode", &component.shadow_mode, shadow_mode, 4) || update;
		update = ImGui::Combo("Sample method", &component.sample_method, sample_method, 2) || update;
		update = ImGui::DragInt("Samples", &component.filter_sample, 0.1f, 0, std::numeric_limits<int32_t>::max()) || update;
		update = ImGui::DragFloat("Scale", &component.filter_scale, 0.01f, 0.f, std::numeric_limits<float>::max(), "%.2f") || update;

		return update;
	});
}

template <>
inline bool draw_component<cmpt::SpotLight>(Entity entity)
{
	return draw_component<cmpt::SpotLight>("Spot Light", entity, [](cmpt::SpotLight &component) {
		const char *const shadow_mode[]   = {"No Shadow", "Hard Shadow", "PCF", "PCSS"};
		const char *const sample_method[] = {"Uniform", "Possion"};

		bool update = false;

		update = ImGui::DragFloat("Intensity", &component.intensity, 0.01f, 0.f, std::numeric_limits<float>::max(), "%.3f") || update;
		update = ImGui::ColorEdit3("Color", glm::value_ptr(component.color)) || update;
		update = ImGui::DragFloat("Cut off", &component.cut_off, 0.0001f, 0.f, 1.f, "%.5f") || update;
		update = ImGui::DragFloat("Outer cut off", &component.outer_cut_off, 0.0001f, 0.f, component.cut_off, "%.5f") || update;
		update = ImGui::DragFloat("Light size", &component.light_size, 0.01f, 0.f, std::numeric_limits<float>::max(), "%.2f") || update;
		ImGui::Text("Shadow Setting");
		update = ImGui::Combo("Shadow Mode", &component.shadow_mode, shadow_mode, 4) || update;
		update = ImGui::Combo("Sample method", &component.sample_method, sample_method, 2) || update;
		update = ImGui::DragInt("Samples", &component.filter_sample, 0.1f, 0, std::numeric_limits<int32_t>::max()) || update;
		update = ImGui::DragFloat("Scale", &component.filter_scale, 0.01f, 0.f, std::numeric_limits<float>::max(), "%.2f") || update;

		return update;
	});
}

template <>
inline bool draw_component<cmpt::PointLight>(Entity entity)
{
	return draw_component<cmpt::PointLight>("Point Light", entity, [](cmpt::PointLight &component) {
		const char *const shadow_mode[]   = {"No Shadow", "Hard Shadow", "PCF", "PCSS"};
		const char *const sample_method[] = {"Uniform", "Possion"};

		bool update = false;

		update = ImGui::DragFloat("Intensity", &component.intensity, 0.01f, 0.f, std::numeric_limits<float>::max(), "%.3f") || update;
		update = ImGui::ColorEdit3("Color", glm::value_ptr(component.color)) || update;
		update = ImGui::DragFloat("Constant", &component.constant, 0.0001f, 0.f, std::numeric_limits<float>::max(), "%.5f") || update;
		update = ImGui::DragFloat("Linear", &component.linear, 0.0001f, 0.f, std::numeric_limits<float>::max(), "%.5f") || update;
		update = ImGui::DragFloat("Quadratic", &component.quadratic, 0.0001f, 0.f, std::numeric_limits<float>::max(), "%.5f") || update;
		update = ImGui::DragFloat("Light size", &component.light_size, 0.01f, 0.f, std::numeric_limits<float>::max(), "%.2f") || update;
		ImGui::Text("Shadow Setting");
		update = ImGui::Combo("Shadow Mode", &component.shadow_mode, shadow_mode, 4) || update;
		update = ImGui::Combo("Sample method", &component.sample_method, sample_method, 2) || update;
		update = ImGui::DragInt("Samples", &component.filter_sample, 0.1f, 0, std::numeric_limits<int32_t>::max()) || update;
		update = ImGui::DragFloat("Scale", &component.filter_scale, 0.01f, 0.f, std::numeric_limits<float>::max(), "%.2f") || update;

		return update;
	});
}

template <>
inline bool draw_component<cmpt::AreaLight>(Entity entity)
{
	return draw_component<cmpt::AreaLight>("Area Light", entity, [](cmpt::AreaLight &component) {
		const char *const shape[] = {"Rectangle", "Ellipse"};

		bool update = false;

		update = ImGui::DragFloat("Intensity", &component.intensity, 0.01f, 0.f, std::numeric_limits<float>::max(), "%.3f") || update;
		update = ImGui::ColorEdit3("Color", glm::value_ptr(component.color)) || update;
		update = ImGui::Combo("Shape", reinterpret_cast<int *>(&component.shape), shape, 2) || update;

		std::string texture = "";
		for (auto& [name, idx] : Renderer::instance()->getResourceCache().getImages())
		{
			if (idx == component.texture_id)
			{
				texture = name;
				break;
			}
		}
		ImGui::Text("Texture");
		update = draw_texture(texture, "Texture") || update;
		component.texture_id = Renderer::instance()->getResourceCache().imageID(texture);

		return update;
	});
}

template <>
inline bool draw_component<cmpt::PerspectiveCamera>(Entity entity)
{
	return draw_component<cmpt::PerspectiveCamera>("Perspective Camera", entity, [entity](cmpt::PerspectiveCamera &component) {
		bool update = false;

		update      = ImGui::DragFloat("Aspect", &component.aspect, 0.01f, 0.f, std::numeric_limits<float>::max(), "%.3f") || update;
		update      = ImGui::DragFloat("Fov", &component.fov, 0.01f, 0.f, 90.f, "%.3f") || update;
		update      = ImGui::DragFloat("Near Plane", &component.near_plane, 0.01f, 0.f, std::numeric_limits<float>::max(), "%.3f") || update;
		update      = ImGui::DragFloat("Far Plane", &component.far_plane, 0.01f, 0.f, std::numeric_limits<float>::max(), "%.3f") || update;
		bool select = entity == Renderer::instance()->Main_Camera;
		if (ImGui::Checkbox("Main Camera", &select))
		{
			if (select)
			{
				Renderer::instance()->Main_Camera = entity;

				auto extent      = Renderer::instance()->getViewportExtent();
				component.aspect = static_cast<float>(extent.width) / static_cast<float>(extent.height);
			}
			else
			{
				Renderer::instance()->Main_Camera = Entity();
			}

			update = true;
		}

		return update;
	});
}

template <>
inline bool draw_component<cmpt::OrthographicCamera>(Entity entity)
{
	return draw_component<cmpt::OrthographicCamera>("Orthographic Camera", entity, [entity](cmpt::OrthographicCamera &component) {
		bool update = false;

		update      = ImGui::DragFloat("Left", &component.left, 0.01f, -std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), "%.3f") || update;
		update      = ImGui::DragFloat("Right", &component.right, 0.01f, -std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), "%.3f") || update;
		update      = ImGui::DragFloat("Bottom", &component.bottom, 0.01f, -std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), "%.3f") || update;
		update      = ImGui::DragFloat("Top", &component.top, 0.01f, -std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), "%.3f") || update;
		update      = ImGui::DragFloat("Near Plane", &component.near_plane, 0.01f, 0.f, std::numeric_limits<float>::max(), "%.3f") || update;
		update      = ImGui::DragFloat("Far Plane", &component.far_plane, 0.01f, 0.f, std::numeric_limits<float>::max(), "%.3f") || update;
		bool select = entity == Renderer::instance()->Main_Camera;
		if (ImGui::Checkbox("Main Camera", &select))
		{
			Renderer::instance()->Main_Camera = select ? entity : Entity();
			update                            = true;
		}

		return update;
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
	ImGui::PopItemWidth();

	if (ImGui::BeginPopup("AddComponent"))
	{
		add_component<cmpt::Light, cmpt::DirectionalLight, cmpt::PointLight, cmpt::SpotLight, cmpt::AreaLight>();
		add_component<cmpt::Renderable, cmpt::StaticMeshRenderer, cmpt::DynamicMeshRenderer, cmpt::CurveRenderer, cmpt::SurfaceRenderer>();
		add_component<cmpt::Camera, cmpt::PerspectiveCamera, cmpt::OrthographicCamera>();
		ImGui::EndPopup();
	}

	if (draw_component<cmpt::Transform, cmpt::Hierarchy, cmpt::StaticMeshRenderer, cmpt::DynamicMeshRenderer, cmpt::CurveRenderer, cmpt::SurfaceRenderer,
	                   cmpt::DirectionalLight, cmpt::PointLight, cmpt::SpotLight, cmpt::AreaLight, cmpt::PerspectiveCamera, cmpt::OrthographicCamera>(entity))
	{
		auto &camera_entity = Renderer::instance()->Main_Camera;

		if (camera_entity && (camera_entity.hasComponent<cmpt::PerspectiveCamera>() || camera_entity.hasComponent<cmpt::OrthographicCamera>()))
		{
			cmpt::Camera *camera = camera_entity.hasComponent<cmpt::PerspectiveCamera>() ?
			                           static_cast<cmpt::Camera *>(&camera_entity.getComponent<cmpt::PerspectiveCamera>()) :
                                       static_cast<cmpt::Camera *>(&camera_entity.getComponent<cmpt::OrthographicCamera>());

			camera->frame_count = 0;
		}
	}

	ImGui::End();
}
}        // namespace Ilum::panel