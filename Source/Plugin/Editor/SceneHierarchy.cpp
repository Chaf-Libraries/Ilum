#include <Scene/Components/AllComponents.hpp>
#include <Editor/Editor.hpp>
#include <Editor/Widget.hpp>
#include <Renderer/Renderer.hpp>
#include <Scene/Node.hpp>
#include <Scene/Scene.hpp>

#include <imgui.h>

using namespace Ilum;

class SceneHierarchy : public Widget
{
  public:
	SceneHierarchy(Editor *editor) :
	    Widget("Scene Hierarchy", editor)
	{
	}

	virtual ~SceneHierarchy() override = default;

	virtual void Tick() override
	{
		if (!ImGui::Begin(m_name.c_str()))
		{
			ImGui::End();
			return;
		}

		if (ImGui::BeginDragDropTarget())
		{
			if (const auto *pay_load = ImGui::AcceptDragDropPayload("Node"))
			{
				if (pay_load->DataSize == sizeof(Node *))
				{
					(*static_cast<Node **>(pay_load->Data))->SetParent(nullptr);
				}
			}
		}

		auto roots = p_editor->GetRenderer()->GetScene()->GetRoots();

		for (auto *root : roots)
		{
			DrawNode(root);
		}

		ImGui::End();
	}

	void DrawNode(Node *node)
	{
		bool has_child = !node->GetChildren().empty();

		bool update = false;

		// Setting up
		ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(5, 5));
		ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_FramePadding | (p_editor->GetSelectedNode() == node ? ImGuiTreeNodeFlags_Selected : 0) | (has_child ? 0 : ImGuiTreeNodeFlags_Leaf);

		ImGui::PushID(static_cast<int32_t>((uint64_t) node));
		bool open = ImGui::TreeNodeEx(std::to_string((uint64_t) node).c_str(), flags, "%s", node->GetName().empty() ? " " : node->GetName().c_str());

		ImGui::PopStyleVar();

		// Delete node
		bool node_deleted = false;
		if (ImGui::BeginPopupContextItem(std::to_string((uint64_t) node).c_str()))
		{
			if (ImGui::MenuItem("Delete Entity"))
			{
				node_deleted = true;
				update       = true;
			}
			ImGui::EndPopup();
		}
		ImGui::PopID();

		// Select entity
		if (ImGui::IsItemClicked())
		{
			p_editor->SelectNode(node);
		}

		// Drag and drop
		if (ImGui::BeginDragDropSource())
		{
			ImGui::SetDragDropPayload("Node", &node, sizeof(Node *));
			ImGui::EndDragDropSource();
		}

		if (ImGui::BeginDragDropTarget())
		{
			if (const auto *pay_load = ImGui::AcceptDragDropPayload("Node"))
			{
				if (pay_load->DataSize == sizeof(Node *))
				{
					Node *new_son = *static_cast<Node **>(pay_load->Data);
					if (node->GetParent() != new_son)
					{
						new_son->SetParent(node);
						new_son->GetComponent<Cmpt::Transform>()->SetDirty();
						update = true;
					}
				}
			}
			ImGui::EndDragDropTarget();
		}

		// Recursively open children nodes
		if (open)
		{
			for (auto *child : node->GetChildren())
			{
				DrawNode(child);
			}
			ImGui::TreePop();
		}

		// Delete callback
		if (node_deleted)
		{
			auto *scene = p_editor->GetRenderer()->GetScene();

			if ((node->HasComponent<Cmpt::OrthographicCamera>() &&node->GetComponent<Cmpt::OrthographicCamera>()==p_editor->GetMainCamera())||
			    (node->HasComponent<Cmpt::PerspectiveCamera>() && node->GetComponent<Cmpt::PerspectiveCamera>() == p_editor->GetMainCamera()))
			{
				p_editor->SetMainCamera();
			}

			if (p_editor->GetSelectedNode() == node)
			{
				p_editor->SelectNode();
			}

			scene->EraseNode(node);
		}
	}
};

extern "C"
{
	__declspec(dllexport) SceneHierarchy *Create(Editor *editor, ImGuiContext *context)
	{
		ImGui::SetCurrentContext(context);
		return new SceneHierarchy(editor);
	}
}