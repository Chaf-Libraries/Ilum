#include <Components/AllComponents.hpp>
#include <Editor/Editor.hpp>
#include <Editor/Widget.hpp>
#include <SceneGraph/Node.hpp>

#include <imgui.h>
#include <imgui_internal.h>

#include <typeindex>
#include <unordered_set>

using namespace Ilum;

class SceneInspector : public Widget
{
  public:
	SceneInspector(Editor *editor) :
	    Widget("Scene Inspector", editor)
	{
	}

	virtual ~SceneInspector() override = default;

	virtual void Tick() override
	{
		auto *select = p_editor->GetSelectedNode();

		ImGui::Begin(m_name.c_str());

		if (!select)
		{
			ImGui::End();
			return;
		}

		// Name
		{
			char buf[64] = {0};
			std::memcpy(buf, select->GetName().data(), sizeof(buf));
			ImGui::PushItemWidth(150.f);
			if (ImGui::InputText("##Tag", buf, sizeof(buf)))
			{
				select->SetName(buf);
			}
			ImGui::PopItemWidth();
		}

		ImGui::SameLine();

		// Add Components
		ImGui::PushItemWidth(-1);
		if (ImGui::Button("Add Component"))
		{
			ImGui::OpenPopup("AddComponent");
		}
		ImGui::PopItemWidth();

		if (ImGui::BeginPopup("AddComponent"))
		{
			AddComponents<Cmpt::Light, Cmpt::PointLight, Cmpt::SpotLight, Cmpt::DirectionalLight, Cmpt::PolygonLight>();
			ImGui::EndPopup();
		}

		for (auto &[type, cmpt] : select->GetComponents())
		{
			const ImGuiTreeNodeFlags tree_node_flags = ImGuiTreeNodeFlags_DefaultOpen | ImGuiTreeNodeFlags_Framed | ImGuiTreeNodeFlags_SpanAvailWidth | ImGuiTreeNodeFlags_AllowItemOverlap | ImGuiTreeNodeFlags_FramePadding;

			ImVec2 content_region_available = ImGui::GetContentRegionAvail();

			ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2{4, 4});
			float line_height = GImGui->Font->FontSize + GImGui->Style.FramePadding.y * 2.0f;
			bool  open        = ImGui::TreeNodeEx((void *) type.hash_code(), tree_node_flags, cmpt->GetName());
			ImGui::PopStyleVar();

			bool remove_component = false;
			if (cmpt->GetType() != typeid(Cmpt::Transform))
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
				cmpt->OnImGui();
				ImGui::TreePop();
			}

			if (remove_component)
			{
				select->EraseComponent(type);
				update = true;
				break;
			}
		}

		ImGui::End();
	}

	template <typename _Ty>
	bool HasComponent()
	{
		return p_editor->GetSelectedNode()->HasComponent<_Ty>();
	}

	template <typename _Ty1, typename... _TyN>
	bool HasComponent()
	{
		return HasComponent<_Ty1>() || HasComponent<_TyN...>();
	}

	template <typename _Ty>
	void AddComponent()
	{
		if (ImGui::MenuItem(typeid(_Ty).name()))
		{
			p_editor->GetSelectedNode()->AddComponent(std::make_unique<_Ty>(p_editor->GetSelectedNode()));
			ImGui::CloseCurrentPopup();
		}
	}

	template <typename _Ty1, typename _Ty2, typename... _TyN>
	void AddComponent()
	{
		AddComponent<_Ty1>();
		AddComponent<_Ty2, _TyN...>();
	}

	template <typename _Base, typename... _TyN>
	void AddComponents()
	{
		if (ImGui::BeginMenu(typeid(_Base).name()))
		{
			AddComponent<_TyN...>();
			ImGui::EndMenu();
		}
	}
};

extern "C"
{
	__declspec(dllexport) SceneInspector *Create(Editor *editor, ImGuiContext *context)
	{
		ImGui::SetCurrentContext(context);
		ConfigureImGui(context);
		return new SceneInspector(editor);
	}
}