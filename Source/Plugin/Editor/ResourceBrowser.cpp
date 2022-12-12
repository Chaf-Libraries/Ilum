#include <Editor/Editor.hpp>
#include <Editor/Widget.hpp>
#include <Renderer/Renderer.hpp>
#include <Resource/Resource.hpp>
#include <Resource/ResourceManager.hpp>

#include <imgui.h>
#include <imgui_internal.h>

using namespace Ilum;

class ResourceBrowser : public Widget
{
  public:
	ResourceBrowser(Editor *editor) :
	    Widget("Resource Browser", editor)
	{
	}

	virtual ~ResourceBrowser() = default;

	template <ResourceType _Ty>
	inline void DrawResource(ResourceManager *manager, float button_size)
	{
		float width = 0.f;

		ImGuiStyle &style    = ImGui::GetStyle();
		style.ItemSpacing    = ImVec2(10.f, 10.f);
		float window_visible = ImGui::GetWindowPos().x + ImGui::GetWindowContentRegionMax().x;

		const std::vector<std::string> resources = manager->GetResources<_Ty>();

		for (const auto &resource : resources)
		{
			ImGui::PushID(resource.c_str());
			ImGui::ImageButton(ImGui::GetIO().Fonts->TexID, ImVec2{button_size, button_size});

			// Drag&Drop source
			if (ImGui::BeginDragDropSource())
			{
				ImGui::SetDragDropPayload(m_resource_types.at(_Ty), resource.c_str(), resource.length() + 1);
				ImGui::EndDragDropSource();
			}

			if (ImGui::BeginPopupContextItem(resource.c_str()))
			{
				if (ImGui::MenuItem("Delete"))
				{
					// manager->EraseResource<_Ty>(uuid);
					ImGui::EndPopup();
					ImGui::PopID();
					return;
				}
				ImGui::EndPopup();
			}
			else if (ImGui::IsItemHovered() && ImGui::IsWindowFocused())
			{
				ImVec2 pos = ImGui::GetIO().MousePos;
				// ImGui::SetNextWindowPos(ImVec2(pos.x + 10.f, pos.y + 10.f));
				//  ImGui::Begin(uuid_str.c_str(), NULL, ImGuiWindowFlags_Tooltip | ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoTitleBar);
				//  ImGui::Text("%s", manager->GetResourceMeta<_Ty>(uuid).c_str());
				//  ImGui::End();
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

	virtual void Tick() override
	{
		if (!ImGui::Begin(m_name.c_str()))
		{
			ImGui::End();
			return;
		}

		ImGui::Columns(2);
		ImGui::SetColumnWidth(0, 200.f);

		if (ImGui::Button("Import"))
		{
		}

		for (auto &[type, name] : m_resource_types)
		{
			bool open = ImGui::TreeNodeEx(name, ImGuiTreeNodeFlags_FramePadding | ImGuiTreeNodeFlags_DefaultOpen | (m_current_type == type ? ImGuiTreeNodeFlags_Selected : 0) | ImGuiTreeNodeFlags_Leaf);
			if (ImGui::IsItemClicked())
			{
				m_current_type = type;
			}

			if (open)
			{
				ImGui::TreePop();
			}
		}

		ImGui::NextColumn();

		switch (m_current_type)
		{
			case ResourceType::Prefab:
				DrawResource<ResourceType::Prefab>(p_editor->GetRenderer()->GetResourceManager(), 100.f);
				break;
			case ResourceType::Mesh:
				DrawResource<ResourceType::Mesh>(p_editor->GetRenderer()->GetResourceManager(), 100.f);
				break;
			case ResourceType::SkinnedMesh:
				DrawResource<ResourceType::SkinnedMesh>(p_editor->GetRenderer()->GetResourceManager(), 100.f);
				break;
			case ResourceType::Texture2D:
				DrawResource<ResourceType::Texture2D>(p_editor->GetRenderer()->GetResourceManager(), 100.f);
				break;
			case ResourceType::Animation:
				DrawResource<ResourceType::Animation>(p_editor->GetRenderer()->GetResourceManager(), 100.f);
				break;
			default:
				break;
		}

		ImGui::End();
	}

  private:
	ResourceType m_current_type = ResourceType::Prefab;

	std::unordered_map<ResourceType, const char *const> m_resource_types = {
	    {ResourceType::Mesh, "Mesh"},
	    {ResourceType::SkinnedMesh, "SkinnedMesh"},
	    {ResourceType::Prefab, "Prefab"},
	    {ResourceType::Texture2D, "Texture2D"},
	    {ResourceType::Animation, "Animation"},
	};
};

extern "C"
{
	EXPORT_API ResourceBrowser *Create(Editor *editor, ImGuiContext *context)
	{
		ImGui::SetCurrentContext(context);
		return new ResourceBrowser(editor);
	}
}