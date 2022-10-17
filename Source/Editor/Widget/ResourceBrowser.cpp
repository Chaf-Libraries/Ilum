#include "ResourceBrowser.hpp"
#include "Editor/Editor.hpp"
#include "ImGui/ImGuiHelper.hpp"

#include <Core/Path.hpp>
#include <Renderer/Renderer.hpp>
#include <Resource/ResourceManager.hpp>

#include <imgui.h>

#pragma warning(push, 0)
#include <nfd.h>
#pragma warning(pop)

namespace Ilum
{
template <ResourceType _Ty>
bool IsResourceFile(const std::string &extension)
{
	return false;
}

template <>
bool IsResourceFile<ResourceType::Texture>(const std::string &extension)
{
	return extension == ".jpg" ||
	       extension == ".png" ||
	       extension == ".bmp" ||
	       extension == ".jpeg" ||
	       extension == ".dds";
}

template <>
bool IsResourceFile<ResourceType::Model>(const std::string &extension)
{
	return extension == ".gltf" ||
	       extension == ".obj" ||
	       extension == ".fbx" ||
	       extension == ".ply" ||
	       extension == ".glb";
}

template <>
bool IsResourceFile<ResourceType::RenderGraph>(const std::string &extension)
{
	return extension == ".rg";
}

template <>
bool IsResourceFile<ResourceType::Scene>(const std::string &extension)
{
	return extension == ".scene";
}

template <>
bool IsResourceFile<ResourceType::Material>(const std::string &extension)
{
	return extension == ".mat";
}

template <ResourceType _Ty>
inline void DrawResource(ResourceManager *manager, float button_size)
{
	float width = 0.f;

	ImGuiStyle &style    = ImGui::GetStyle();
	style.ItemSpacing    = ImVec2(10.f, 10.f);
	float window_visible = ImGui::GetWindowPos().x + ImGui::GetWindowContentRegionMax().x;

	for (auto &uuid : manager->GetResourceUUID<_Ty>())
	{
		ImGui::PushID(&uuid);
		ImGui::ImageButton(manager->GetThumbnail<_Ty>(), ImVec2{button_size, button_size});

		// Drag&Drop source
		if (ImGui::BeginDragDropSource())
		{
			const size_t uuid_ = uuid;
			
			ImGui::SetDragDropPayload(rttr::type::get<ResourceType>().get_enumeration().value_to_name(_Ty).to_string().c_str(), &uuid_, sizeof(size_t));
			ImGui::EndDragDropSource();
		}

		std::string uuid_str = std::to_string(uuid);
		if (ImGui::BeginPopupContextItem(uuid_str.c_str()))
		{
			if (ImGui::MenuItem("Delete"))
			{
				manager->EraseResource<_Ty>(uuid);
				ImGui::EndPopup();
				ImGui::PopID();
				return;
			}
			ImGui::EndPopup();
		}
		else if (ImGui::IsItemHovered() && ImGui::IsWindowFocused())
		{
			ImVec2 pos = ImGui::GetIO().MousePos;
			ImGui::SetNextWindowPos(ImVec2(pos.x + 10.f, pos.y + 10.f));
			ImGui::Begin(uuid_str.c_str(), NULL, ImGuiWindowFlags_Tooltip | ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoTitleBar);
			ImGui::Text("%s", manager->GetResourceMeta<_Ty>(uuid).c_str());
			ImGui::End();
		}

		float last_button = ImGui::GetItemRectMax().x;
		float next_button = last_button + style.ItemSpacing.x + button_size;
		if (next_button < window_visible)
		{
			ImGui::SameLine();
		}

		ImGui::PopID();
	}
}

template <>
inline void DrawResource<ResourceType::Texture>(ResourceManager *manager, float button_size)
{
	float width = 0.f;

	ImGuiStyle &style    = ImGui::GetStyle();
	style.ItemSpacing    = ImVec2(10.f, 10.f);
	float window_visible = ImGui::GetWindowPos().x + ImGui::GetWindowContentRegionMax().x;

	for (auto &uuid : manager->GetResourceUUID<ResourceType::Texture>())
	{
		ImGui::PushID(&uuid);

		if (manager->IsValid<ResourceType::Texture>(uuid))
		{
			ImGui::ImageButton(manager->GetResource<ResourceType::Texture>(uuid)->GetTexture(), ImVec2{button_size, button_size});
		}
		else
		{
			ImGui::ImageButton(manager->GetThumbnail<ResourceType::Texture>(), ImVec2{button_size, button_size});
		}

		// Drag&Drop source
		if (ImGui::BeginDragDropSource())
		{
			const size_t uuid_ = uuid;
			ImGui::SetDragDropPayload(rttr::type::get<ResourceType>().get_enumeration().value_to_name(ResourceType::Texture).to_string().c_str(), &uuid_, sizeof(size_t));
			ImGui::EndDragDropSource();
		}

		std::string uuid_str = std::to_string(uuid);
		if (ImGui::BeginPopupContextItem(uuid_str.c_str()))
		{
			if (ImGui::MenuItem("Delete"))
			{
				manager->EraseResource<ResourceType::Texture>(uuid);
				ImGui::EndPopup();
				ImGui::PopID();
				return;
			}
			ImGui::EndPopup();
		}
		else if (ImGui::IsItemHovered() && ImGui::IsWindowFocused())
		{
			ImVec2 pos = ImGui::GetIO().MousePos;
			ImGui::SetNextWindowPos(ImVec2(pos.x + 10.f, pos.y + 10.f));
			ImGui::Begin(uuid_str.c_str(), NULL, ImGuiWindowFlags_Tooltip | ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoTitleBar);
			ImGui::Text("%s", manager->GetResourceMeta<ResourceType::Texture>(uuid).c_str());
			ImGui::Separator();
			ImGui::End();
		}

		float last_button = ImGui::GetItemRectMax().x;
		float next_button = last_button + style.ItemSpacing.x + button_size;
		if (next_button < window_visible)
		{
			ImGui::SameLine();
		}

		ImGui::PopID();
	}
}

ResourceBrowser::ResourceBrowser(Editor *editor) :
    Widget("Resource Browser", editor)
{
}

ResourceBrowser::~ResourceBrowser()
{
}

void ResourceBrowser::Tick()
{
	auto *resource_manager = p_editor->GetRenderer()->GetResourceManager();

	ImGui::Begin(m_name.c_str());

	if (ImGui::Button("Import"))
	{
		char *path = nullptr;
		if (NFD_OpenDialog("jpg,png,bmp,jpeg,dds,gltf,obj,glb,fbx,scene,rg,mat", Path::GetInstance().GetCurrent(false).c_str(), &path) == NFD_OKAY)
		{
			std::string extension = Path::GetInstance().GetFileExtension(path);
			if (IsResourceFile<ResourceType::Texture>(extension))
			{
				resource_manager->Import<ResourceType::Texture>(path);
			}
			else if (IsResourceFile<ResourceType::Model>(extension))
			{
				resource_manager->Import<ResourceType::Model>(path);
			}
			else if (IsResourceFile<ResourceType::Scene>(extension))
			{
				resource_manager->Import<ResourceType::Scene>(path);
			}
			else if (IsResourceFile<ResourceType::RenderGraph>(extension))
			{
				resource_manager->Import<ResourceType::RenderGraph>(path);
			}
			else if (IsResourceFile<ResourceType::Material>(extension))
			{
				resource_manager->Import<ResourceType::Material>(path);
			}
		}
	}

	ImGui::SameLine();

	auto region_width = ImGui::GetContentRegionAvailWidth();

	static const char  *ASSET_TYPE[] = {"None", "Model", "Texture", "Scene", "RenderGraph", "Material"};
	static ResourceType type         = ResourceType::None;

	ImGui::Combo("Resource Type", reinterpret_cast<int32_t *>(&type), ASSET_TYPE, 6);

	ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(0.f, 0.f));
	ImGui::Separator();
	ImGui::BeginChild("scrolling", ImVec2(0, 0), false, ImGuiWindowFlags_HorizontalScrollbar);

	switch (type)
	{
		case ResourceType::None:
			break;
		case ResourceType::Model:
			DrawResource<ResourceType::Model>(resource_manager, m_button_size);
			break;
		case ResourceType::Texture:
			DrawResource<ResourceType::Texture>(resource_manager, m_button_size);
			break;
		case ResourceType::Scene:
			DrawResource<ResourceType::Scene>(resource_manager, m_button_size);
			break;
		case ResourceType::RenderGraph:
			DrawResource<ResourceType::RenderGraph>(resource_manager, m_button_size);
			break;
		case ResourceType::Material:
			DrawResource<ResourceType::Material>(resource_manager, m_button_size);
			break;
		default:
			break;
	}

	ImGui::EndChild();
	ImGui::PopStyleVar();

	ImGui::End();
}
}        // namespace Ilum