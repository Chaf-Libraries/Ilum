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
			ImGui::SetDragDropPayload(typeid(_Ty).name(), &uuid_, sizeof(size_t));
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
			ImGui::SetDragDropPayload(typeid(ResourceType::Texture).name(), &uuid_, sizeof(size_t));
			ImGui::EndDragDropSource();
			break;
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
		if (NFD_OpenDialog("jpg,png,bmp,jpeg,dds,gltf,obj,glb,fbx,scene,rg", Path::GetInstance().GetCurrent(false).c_str(), &path) == NFD_OKAY)
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
		}
	}

	ImGui::SameLine();

	auto region_width = ImGui::GetContentRegionAvailWidth();

	static const char  *ASSET_TYPE[] = {"None", "Model", "Texture", "Scene", "RenderGraph"};
	static ResourceType type         = ResourceType::None;

	ImGui::Combo("Resource Type", reinterpret_cast<int32_t *>(&type), ASSET_TYPE, 5);

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
		default:
			break;
	}

	ImGui::EndChild();
	ImGui::PopStyleVar();

	ImGui::End();
}

// void ResourceBrowser::DrawTextureBrowser()
//{
//	auto *resource_manager = p_editor->GetRenderer()->GetResourceManager();
//
//	float width = 0.f;
//
//	ImGuiStyle &style    = ImGui::GetStyle();
//	style.ItemSpacing    = ImVec2(10.f, 10.f);
//	float window_visible = ImGui::GetWindowPos().x + ImGui::GetWindowContentRegionMax().x;
//
//	for (auto &meta : resource_manager->GetTextureMeta())
//	{
//		ImGui::ImageButton(meta->thumbnail.get(), ImVec2{m_button_size, m_button_size});
//
//		// Drag&Drop source
//		if (ImGui::BeginDragDropSource())
//		{
//			ImGui::SetDragDropPayload("Texture", &meta->uuid, sizeof(std::string));
//			ImGui::EndDragDropSource();
//		}
//
//		// Image Hint
//		if (ImGui::BeginPopupContextItem(meta->desc.name.c_str()))
//		{
//			if (ImGui::MenuItem("Delete"))
//			{
//				// Renderer::instance()->getResourceCache().removeImage(name);
//			}
//			ImGui::EndPopup();
//		}
//		else if (ImGui::IsItemHovered() && ImGui::IsWindowFocused())
//		{
//			ImVec2 pos = ImGui::GetIO().MousePos;
//			ImGui::SetNextWindowPos(ImVec2(pos.x + 10.f, pos.y + 10.f));
//			ImGui::Begin(meta->desc.name.c_str(), NULL, ImGuiWindowFlags_Tooltip | ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoTitleBar);
//			ImGui::Text(meta->desc.name.c_str());
//			ImGui::Separator();
//			ImGui::Text("format: %s", rttr::type::get_by_name("RHIFormat").get_enumeration().value_to_name(meta->desc.format).data());
//			ImGui::Text("width: %s", std::to_string(meta->desc.width).c_str());
//			ImGui::Text("height: %s", std::to_string(meta->desc.height).c_str());
//			ImGui::Text("mip levels: %s", std::to_string(meta->desc.mips).c_str());
//			ImGui::Text("layers: %s", std::to_string(meta->desc.layers).c_str());
//			ImGui::Text("external: %s", std::to_string(meta->desc.external).c_str());
//			ImGui::End();
//		}
//
//		float last_button = ImGui::GetItemRectMax().x;
//		float next_button = last_button + style.ItemSpacing.x + m_button_size;
//		if (next_button < window_visible)
//		{
//			ImGui::SameLine();
//		}
//	}
// }
//
// void ResourceBrowser::DrawModelBrowser()
//{
//	auto *resource_manager = p_editor->GetRenderer()->GetResourceManager();
//
//	float width = 0.f;
//
//	ImGuiStyle &style    = ImGui::GetStyle();
//	style.ItemSpacing    = ImVec2(10.f, 10.f);
//	float window_visible = ImGui::GetWindowPos().x + ImGui::GetWindowContentRegionMax().x;
//
//	for (auto &meta : resource_manager->GetModelMeta())
//	{
//		ImGui::PushID(&meta);
//		ImGui::ImageButton(resource_manager->GetThumbnail(ResourceType::Model), ImVec2{m_button_size, m_button_size});
//
//		// Drag&Drop source
//		if (ImGui::BeginDragDropSource())
//		{
//			ImGui::SetDragDropPayload("Model", &meta->uuid, sizeof(std::string));
//			ImGui::EndDragDropSource();
//		}
//		ImGui::PopID();
//
//		// Image Hint
//		if (ImGui::BeginPopupContextItem(meta->uuid.c_str()))
//		{
//			if (ImGui::MenuItem("Delete"))
//			{
//				// Renderer::instance()->getResourceCache().removeImage(name);
//			}
//			ImGui::EndPopup();
//		}
//		else if (ImGui::IsItemHovered() && ImGui::IsWindowFocused())
//		{
//			ImVec2 pos = ImGui::GetIO().MousePos;
//			ImGui::SetNextWindowPos(ImVec2(pos.x + 10.f, pos.y + 10.f));
//			ImGui::Begin(meta->uuid.c_str(), NULL, ImGuiWindowFlags_Tooltip | ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoTitleBar);
//			ImGui::Text(meta->name.c_str());
//			ImGui::Text("Vertices count: %u", meta->vertices_count);
//			ImGui::Text("Triangles count: %u", meta->triangle_count);
//			ImGui::Text("Submeshes count: %ld", meta->submeshes.size());
//			ImGui::End();
//		}
//
//		float last_button = ImGui::GetItemRectMax().x;
//		float next_button = last_button + style.ItemSpacing.x + m_button_size;
//		if (next_button < window_visible)
//		{
//			ImGui::SameLine();
//		}
//	}
// }
//
// void ResourceBrowser::DrawSceneBrowser()
//{
//	auto *resource_manager = p_editor->GetRenderer()->GetResourceManager();
//
//	float width = 0.f;
//
//	ImGuiStyle &style    = ImGui::GetStyle();
//	style.ItemSpacing    = ImVec2(10.f, 10.f);
//	float window_visible = ImGui::GetWindowPos().x + ImGui::GetWindowContentRegionMax().x;
//
//	for (auto &[uuid, meta] : resource_manager->GetSceneMeta())
//	{
//		ImGui::ImageButton(resource_manager->GetThumbnail(ResourceType::Scene), ImVec2{m_button_size, m_button_size});
//
//		// Drag&Drop source
//		if (ImGui::BeginDragDropSource())
//		{
//			ImGui::SetDragDropPayload("Scene", &meta->uuid, sizeof(std::string));
//			ImGui::EndDragDropSource();
//		}
//
//		// Image Hint
//		if (ImGui::BeginPopupContextItem(uuid.c_str()))
//		{
//			if (ImGui::MenuItem("Delete"))
//			{
//				// Renderer::instance()->getResourceCache().removeImage(name);
//			}
//			ImGui::EndPopup();
//		}
//		else if (ImGui::IsItemHovered() && ImGui::IsWindowFocused())
//		{
//			ImVec2 pos = ImGui::GetIO().MousePos;
//			ImGui::SetNextWindowPos(ImVec2(pos.x + 10.f, pos.y + 10.f));
//			ImGui::Begin(uuid.c_str(), NULL, ImGuiWindowFlags_Tooltip | ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoTitleBar);
//			ImGui::Text(meta->name.c_str());
//			ImGui::End();
//		}
//
//		float last_button = ImGui::GetItemRectMax().x;
//		float next_button = last_button + style.ItemSpacing.x + m_button_size;
//		if (next_button < window_visible)
//		{
//			ImGui::SameLine();
//		}
//	}
// }
//
// void ResourceBrowser::DrawRenderGraphBrowser()
//{
//	auto *resource_manager = p_editor->GetRenderer()->GetResourceManager();
//
//	float width = 0.f;
//
//	ImGuiStyle &style    = ImGui::GetStyle();
//	style.ItemSpacing    = ImVec2(10.f, 10.f);
//	float window_visible = ImGui::GetWindowPos().x + ImGui::GetWindowContentRegionMax().x;
//
//	for (auto &[uuid, meta] : resource_manager->GetRenderGraphMeta())
//	{
//		ImGui::ImageButton(resource_manager->GetThumbnail(ResourceType::RenderGraph), ImVec2{m_button_size, m_button_size});
//
//		// Drag&Drop source
//		if (ImGui::BeginDragDropSource())
//		{
//			ImGui::SetDragDropPayload("RenderGraph", &meta->uuid, sizeof(std::string));
//			ImGui::EndDragDropSource();
//		}
//
//		// Image Hint
//		if (ImGui::BeginPopupContextItem(uuid.c_str()))
//		{
//			if (ImGui::MenuItem("Delete"))
//			{
//				// Renderer::instance()->getResourceCache().removeImage(name);
//			}
//			ImGui::EndPopup();
//		}
//		else if (ImGui::IsItemHovered() && ImGui::IsWindowFocused())
//		{
//			ImVec2 pos = ImGui::GetIO().MousePos;
//			ImGui::SetNextWindowPos(ImVec2(pos.x + 10.f, pos.y + 10.f));
//			ImGui::Begin(uuid.c_str(), NULL, ImGuiWindowFlags_Tooltip | ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoTitleBar);
//			ImGui::Text(meta->name.c_str());
//			ImGui::End();
//		}
//
//		float last_button = ImGui::GetItemRectMax().x;
//		float next_button = last_button + style.ItemSpacing.x + m_button_size;
//		if (next_button < window_visible)
//		{
//			ImGui::SameLine();
//		}
//	}
// }
}        // namespace Ilum