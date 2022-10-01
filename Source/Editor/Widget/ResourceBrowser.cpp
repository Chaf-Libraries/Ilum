#include "ResourceBrowser.hpp"
#include "Editor/Editor.hpp"

#include <Core/Path.hpp>
#include <Renderer/Renderer.hpp>
#include <Resource/ResourceManager.hpp>

#include <imgui.h>

#pragma warning(push, 0)
#include <nfd.h>
#pragma warning(pop)

namespace Ilum
{
bool IsTextureResource(const std::string &extension)
{
	return extension == ".jpg" ||
	       extension == ".png" ||
	       extension == ".bmp" ||
	       extension == ".jpeg" ||
	       extension == ".dds";
}

bool IsModelResource(const std::string &extension)
{
	return extension == ".gltf";
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
		if (NFD_OpenDialog("jpg,png,bmp,jpeg,dds,gltf", Path::GetInstance().GetCurrent(false).c_str(), &path) == NFD_OKAY)
		{
			std::string extension = Path::GetInstance().GetFileExtension(path);
			if (IsTextureResource(extension))
			{
				resource_manager->ImportTexture(path);
			}
			if (IsModelResource(extension))
			{
				resource_manager->ImportModel(path);
			}
		}
	}

	ImGui::SameLine();

	auto region_width = ImGui::GetContentRegionAvailWidth();

	static const char *ASSET_TYPE[] = {"Texture", "Model", "Scene", "Render Graph"};
	static int32_t     current_item = 0;

	ImGui::Combo("Asset Type", &current_item, ASSET_TYPE, 4);

	ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(0.f, 0.f));
	ImGui::Separator();
	ImGui::BeginChild("scrolling", ImVec2(0, 0), false, ImGuiWindowFlags_HorizontalScrollbar);
	if (current_item == 0)
	{
		DrawTextureBrowser();
	}
	else if (current_item == 1)
	{
		DrawModelBrowser();
	}
	else if (current_item == 2)
	{
		DrawSceneBrowser();
	}
	else if (current_item == 3)
	{
		DrawRenderGraphBrowser();
	}

	ImGui::EndChild();
	ImGui::PopStyleVar();

	ImGui::End();
}

void ResourceBrowser::DrawTextureBrowser()
{
	auto *resource_manager = p_editor->GetRenderer()->GetResourceManager();

	float width = 0.f;

	ImGuiStyle &style    = ImGui::GetStyle();
	style.ItemSpacing    = ImVec2(10.f, 10.f);
	float window_visible = ImGui::GetWindowPos().x + ImGui::GetWindowContentRegionMax().x;

	for (auto &meta : resource_manager->GetTextureMeta())
	{
		ImGui::ImageButton(meta->thumbnail.get(), ImVec2{m_button_size, m_button_size});

		// Drag&Drop source
		if (ImGui::BeginDragDropSource())
		{
			ImGui::SetDragDropPayload("Texture", &meta->uuid, sizeof(std::string));
			ImGui::EndDragDropSource();
		}

		// Image Hint
		if (ImGui::BeginPopupContextItem(meta->desc.name.c_str()))
		{
			if (ImGui::MenuItem("Delete"))
			{
				// Renderer::instance()->getResourceCache().removeImage(name);
			}
			ImGui::EndPopup();
		}
		else if (ImGui::IsItemHovered() && ImGui::IsWindowFocused())
		{
			ImVec2 pos = ImGui::GetIO().MousePos;
			ImGui::SetNextWindowPos(ImVec2(pos.x + 10.f, pos.y + 10.f));
			ImGui::Begin(meta->desc.name.c_str(), NULL, ImGuiWindowFlags_Tooltip | ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoTitleBar);
			ImGui::Text(meta->desc.name.c_str());
			ImGui::Separator();
			ImGui::Text("format: %s", rttr::type::get_by_name("Ilum::RHIFormat").get_enumeration().value_to_name(meta->desc.format).data());
			ImGui::Text("width: %s", std::to_string(meta->desc.width).c_str());
			ImGui::Text("height: %s", std::to_string(meta->desc.height).c_str());
			ImGui::Text("mip levels: %s", std::to_string(meta->desc.mips).c_str());
			ImGui::Text("layers: %s", std::to_string(meta->desc.layers).c_str());
			ImGui::Text("external: %s", std::to_string(meta->desc.external).c_str());
			ImGui::End();
		}

		float last_button = ImGui::GetItemRectMax().x;
		float next_button = last_button + style.ItemSpacing.x + m_button_size;
		if (next_button < window_visible)
		{
			ImGui::SameLine();
		}
	}
}

void ResourceBrowser::DrawModelBrowser()
{
	auto *resource_manager = p_editor->GetRenderer()->GetResourceManager();

	float width = 0.f;

	ImGuiStyle &style    = ImGui::GetStyle();
	style.ItemSpacing    = ImVec2(10.f, 10.f);
	float window_visible = ImGui::GetWindowPos().x + ImGui::GetWindowContentRegionMax().x;

	for (auto &meta : resource_manager->GetModelMeta())
	{
		ImGui::PushID(&meta);
		ImGui::ImageButton(resource_manager->GetThumbnail(ResourceType::Model), ImVec2{m_button_size, m_button_size});

		// Drag&Drop source
		if (ImGui::BeginDragDropSource())
		{
			ImGui::SetDragDropPayload("Model", &meta->uuid, sizeof(std::string));
			ImGui::EndDragDropSource();
		}
		ImGui::PopID();

		// Image Hint
		if (ImGui::BeginPopupContextItem(meta->uuid.c_str()))
		{
			if (ImGui::MenuItem("Delete"))
			{
				// Renderer::instance()->getResourceCache().removeImage(name);
			}
			ImGui::EndPopup();
		}
		else if (ImGui::IsItemHovered() && ImGui::IsWindowFocused())
		{
			ImVec2 pos = ImGui::GetIO().MousePos;
			ImGui::SetNextWindowPos(ImVec2(pos.x + 10.f, pos.y + 10.f));
			ImGui::Begin(meta->uuid.c_str(), NULL, ImGuiWindowFlags_Tooltip | ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoTitleBar);
			ImGui::Text(meta->name.c_str());
			ImGui::Text("Vertices count: %u", meta->vertices_count);
			ImGui::Text("Triangles count: %u", meta->triangle_count);
			ImGui::Text("Submeshes count: %ld", meta->submeshes.size());
			ImGui::End();
		}

		float last_button = ImGui::GetItemRectMax().x;
		float next_button = last_button + style.ItemSpacing.x + m_button_size;
		if (next_button < window_visible)
		{
			ImGui::SameLine();
		}
	}
}

void ResourceBrowser::DrawSceneBrowser()
{
	auto *resource_manager = p_editor->GetRenderer()->GetResourceManager();

	float width = 0.f;

	ImGuiStyle &style    = ImGui::GetStyle();
	style.ItemSpacing    = ImVec2(10.f, 10.f);
	float window_visible = ImGui::GetWindowPos().x + ImGui::GetWindowContentRegionMax().x;

	for (auto &[uuid, meta] : resource_manager->GetSceneMeta())
	{
		ImGui::ImageButton(resource_manager->GetThumbnail(ResourceType::Scene), ImVec2{m_button_size, m_button_size});

		// Drag&Drop source
		if (ImGui::BeginDragDropSource())
		{
			ImGui::SetDragDropPayload("Scene", &meta->uuid, sizeof(std::string));
			ImGui::EndDragDropSource();
		}

		// Image Hint
		if (ImGui::BeginPopupContextItem(uuid.c_str()))
		{
			if (ImGui::MenuItem("Delete"))
			{
				// Renderer::instance()->getResourceCache().removeImage(name);
			}
			ImGui::EndPopup();
		}
		else if (ImGui::IsItemHovered() && ImGui::IsWindowFocused())
		{
			ImVec2 pos = ImGui::GetIO().MousePos;
			ImGui::SetNextWindowPos(ImVec2(pos.x + 10.f, pos.y + 10.f));
			ImGui::Begin(uuid.c_str(), NULL, ImGuiWindowFlags_Tooltip | ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoTitleBar);
			ImGui::Text(meta->name.c_str());
			ImGui::End();
		}

		float last_button = ImGui::GetItemRectMax().x;
		float next_button = last_button + style.ItemSpacing.x + m_button_size;
		if (next_button < window_visible)
		{
			ImGui::SameLine();
		}
	}
}

void ResourceBrowser::DrawRenderGraphBrowser()
{
	auto *resource_manager = p_editor->GetRenderer()->GetResourceManager();

	float width = 0.f;

	ImGuiStyle &style    = ImGui::GetStyle();
	style.ItemSpacing    = ImVec2(10.f, 10.f);
	float window_visible = ImGui::GetWindowPos().x + ImGui::GetWindowContentRegionMax().x;

	for (auto &[uuid, meta] : resource_manager->GetRenderGraphMeta())
	{
		ImGui::ImageButton(resource_manager->GetThumbnail(ResourceType::RenderGraph), ImVec2{m_button_size, m_button_size});

		// Drag&Drop source
		if (ImGui::BeginDragDropSource())
		{
			ImGui::SetDragDropPayload("RenderGraph", &meta->uuid, sizeof(std::string));
			ImGui::EndDragDropSource();
		}

		// Image Hint
		if (ImGui::BeginPopupContextItem(uuid.c_str()))
		{
			if (ImGui::MenuItem("Delete"))
			{
				// Renderer::instance()->getResourceCache().removeImage(name);
			}
			ImGui::EndPopup();
		}
		else if (ImGui::IsItemHovered() && ImGui::IsWindowFocused())
		{
			ImVec2 pos = ImGui::GetIO().MousePos;
			ImGui::SetNextWindowPos(ImVec2(pos.x + 10.f, pos.y + 10.f));
			ImGui::Begin(uuid.c_str(), NULL, ImGuiWindowFlags_Tooltip | ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoTitleBar);
			ImGui::Text(meta->name.c_str());
			ImGui::End();
		}

		float last_button = ImGui::GetItemRectMax().x;
		float next_button = last_button + style.ItemSpacing.x + m_button_size;
		if (next_button < window_visible)
		{
			ImGui::SameLine();
		}
	}
}
}        // namespace Ilum