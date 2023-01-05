#include <Editor/Editor.hpp>
#include <Editor/Widget.hpp>
#include <Material/MaterialGraph.hpp>
#include <Renderer/Renderer.hpp>
#include <Resource/Resource/Material.hpp>
#include <Resource/ResourceManager.hpp>

#include <imgui.h>
#include <imgui_internal.h>
#include <imnodes/imnodes.h>

#include <nfd.h>

using namespace Ilum;

class MaterialGraphEditor : public Widget
{
  public:
	MaterialGraphEditor(Editor *editor) :
	    Widget("Material Editor", editor)
	{
		ImNodes::CreateContext();
		m_context = ImNodes::EditorContextCreate();
	}

	virtual ~MaterialGraphEditor() override
	{
		ImNodes::DestroyContext();
		ImNodes::EditorContextFree(m_context);
	}

	virtual void Tick() override
	{
		if (!ImGui::Begin(m_name.c_str()))
		{
			ImGui::End();
			return;
		}

		auto *resource_manager = p_editor->GetRenderer()->GetResourceManager();
		auto *resource         = resource_manager->Get<ResourceType::Material>(m_material_name);

		ImGui::Columns(2);

		// Inspector
		{
			ImGui::BeginChild("Material Editor Inspector", ImVec2(0, 0), false, ImGuiWindowFlags_HorizontalScrollbar);

			ImGui::Text("Material Editor Inspector");

			{
				ImGui::PushItemWidth(100.f);
				char buf[128] = {0};
				std::memcpy(buf, m_material_name.data(), sizeof(buf));
				if (ImGui::InputText("##NewMaterial", buf, sizeof(buf)))
				{
					m_material_name = buf;
				}
				ImGui::PopItemWidth();

				ImGui::SameLine();

				if (!m_material_name.empty() && !resource)
				{
					if (ImGui::Button("New Material"))
					{
						MaterialGraphDesc desc;
						desc.SetName(m_material_name);
						resource_manager->Add<ResourceType::Material>(p_editor->GetRHIContext(), m_material_name, std::move(desc));
						resource = resource_manager->Get<ResourceType::Material>(m_material_name);
					}
				}
				else
				{
					ImGui::Text("Material Name");
				}
			}

			if (ImGui::GetScrollY() >= ImGui::GetScrollMaxY())
			{
				ImGui::SetScrollHereY(1.0f);
			}

			ImGui::EndChild();
		}

		ImGui::NextColumn();

		ImGui::BeginChild("Material Graph Editor", ImVec2(0, 0), false, ImGuiWindowFlags_MenuBar);

		MenuBar();

		HandleSelection();

		ImNodes::BeginNodeEditor();

		if (resource)
		{
			PopupWindow(resource);
			DrawNodes(resource);
			DrawEdges(resource);
		}

		ImNodes::MiniMap(0.1f);
		ImNodes::EndNodeEditor();

		DragDropResource();

		if (resource)
		{
			AddEdge(resource);
		}

		ImGui::EndChild();

		ImGui::End();
	}

	void DragDropResource()
	{
		if (ImGui::BeginDragDropTarget())
		{
			if (const auto *pay_load = ImGui::AcceptDragDropPayload("Material"))
			{
				m_material_name = static_cast<const char *>(pay_load->Data);
				auto *resource  = p_editor->GetRenderer()->GetResourceManager()->Get<ResourceType::Material>(m_material_name);
				if (resource)
				{
					ImNodes::LoadCurrentEditorStateFromIniString(resource->GetLayout().data(), resource->GetLayout().length());
					auto &desc = resource->GetDesc();
					for (auto &[node_handle, node] : desc.GetNodes())
					{
						m_current_handle = glm::max(m_current_handle, node_handle + 1);
						for (auto &[pin_handle, pin] : node.GetPins())
						{
							m_current_handle = glm::max(m_current_handle, pin_handle + 1);
						}
					}
				}
			}
		}
	}

	void MenuBar()
	{
		auto *resource = p_editor->GetRenderer()->GetResourceManager()->Get<ResourceType::Material>(m_material_name);

		if (!resource)
		{
			return;
		}

		if (ImGui::BeginMenuBar())
		{
			if (ImGui::BeginMenu("Node"))
			{
				for (const auto &file : std::filesystem::directory_iterator("shared/Material/"))
				{
					std::string filename = file.path().filename().string();
					{
						size_t pos1 = filename.find_first_of('.');
						size_t pos2 = filename.find_first_of('.', pos1 + 1);
						size_t pos3 = filename.find_first_of('.', pos2 + 1);

						std::string category  = filename.substr(pos1 + 1, pos2 - pos1 - 1);
						std::string node_name = filename.substr(pos2 + 1, pos3 - pos2 - 1);

						if (ImGui::BeginMenu(category.c_str()))
						{
							if (ImGui::MenuItem(node_name.c_str()))
							{
								MaterialNodeDesc desc;
								PluginManager::GetInstance().Call(file.path().string(), "Create", &desc, &m_current_handle);
								resource->GetDesc().AddNode(m_current_handle++, std::move(desc));
							}
							ImGui::EndMenu();
						}
					}
				}
				ImGui::EndMenu();
			}

			if (ImGui::MenuItem("Clear"))
			{
				resource->GetDesc().Clear();
			}

			if (ImGui::MenuItem("Compile"))
			{
				resource->Compile(p_editor->GetRenderer() ->GetRHIContext(), p_editor->GetRenderer()->GetResourceManager(), ImNodes::SaveCurrentEditorStateToIniString());
			}

			ImGui::EndMenuBar();
		}
	}

	void PopupWindow(Resource<ResourceType::Material> *resource)
	{
		if (ImGui::BeginPopupContextWindow(0, ImGuiPopupFlags_MouseButtonRight))
		{
			if (ImGui::MenuItem("Remove"))
			{
				if (!m_select_links.empty() || !m_select_nodes.empty())
				{
					for (auto &link : m_select_links)
					{
						for (auto &[dst, src] : resource->GetDesc().GetEdges())
						{
							if (link == static_cast<int32_t>(Hash(src, dst)))
							{
								resource->GetDesc().EraseLink(static_cast<size_t>(src), static_cast<size_t>(dst));
								break;
							}
						}
					}
					for (auto &node : m_select_nodes)
					{
						resource->GetDesc().EraseNode(node);
					}
				}
			}
			ImGui::EndPopup();
		}
	}

	void EditMaterialNodePin(const MaterialNodePin &pin)
	{
		switch (pin.type)
		{
			case MaterialNodePin::Type::Float:
				ImGui::DragFloat("", pin.variant.Convert<float>(), 0.1f, 0.f, 0.f, "%.1f");
				break;
			case MaterialNodePin::Type::Float3:
				ImGui::DragFloat3("", glm::value_ptr(*pin.variant.Convert<glm::vec3>()), 0.1f, 0.f, 0.f, "%.1f");
				break;
			case MaterialNodePin::Type::RGB:
				ImGui::ColorEdit3("", glm::value_ptr(*pin.variant.Convert<glm::vec3>()), ImGuiColorEditFlags_NoInputs);
				break;
			default:
				break;
		}
	}

	void DrawNodes(Resource<ResourceType::Material> *resource)
	{
		const float node_width = 120.0f;
		ImGui::PushItemWidth(120.f);
		for (auto &[node_handle, node_desc] : resource->GetDesc().GetNodes())
		{
			ImNodes::BeginNode(static_cast<int32_t>(node_handle));
			ImNodes::BeginNodeTitleBar();
			ImGui::Text(node_desc.GetName().c_str());
			ImNodes::EndNodeTitleBar();
			PluginManager::GetInstance().Call(fmt::format("shared/Material/Material.{}.{}.dll", node_desc.GetCategory(), node_desc.GetName()), "OnImGui", &node_desc, p_editor, ImGui::GetCurrentContext());
			for (auto &[pin_handle, pin] : node_desc.GetPins())
			{
				if (!pin.enable)
				{
					continue;
				}

				if (pin.attribute == MaterialNodePin::Attribute::Input)
				{
					ImNodes::PushColorStyle(ImNodesCol_Pin, m_pin_color[pin.type]);
					ImNodes::BeginInputAttribute(static_cast<int32_t>(pin_handle));
					ImGui::TextUnformatted(pin.name.c_str());
					if (!resource->GetDesc().HasLink(pin_handle) && !pin.variant.Empty())
					{
						ImGui::SameLine();
						ImGui::PushItemWidth(node_width - ImGui::CalcTextSize(pin.name.c_str()).x);
						EditMaterialNodePin(pin);
						ImGui::PopItemWidth();
					}
					ImNodes::EndInputAttribute();
					ImNodes::PopColorStyle();
				}
				else
				{
					ImNodes::PushColorStyle(ImNodesCol_Pin, m_pin_color[pin.type]);
					ImNodes::BeginOutputAttribute(static_cast<int32_t>(pin_handle));
					const float label_width = ImGui::CalcTextSize(pin.name.c_str()).x;
					ImGui::Indent(node_width - label_width);
					if (!pin.variant.Empty())
					{
						ImGui::PushItemWidth(node_width - ImGui::CalcTextSize(pin.name.c_str()).x);
						EditMaterialNodePin(pin);
						ImGui::PopItemWidth();
						ImGui::SameLine();
					}
					ImGui::TextUnformatted(pin.name.c_str());
					ImNodes::EndOutputAttribute();
					ImNodes::PopColorStyle();
				}
			}
			ImNodes::EndNode();
		}
		ImGui::PopItemWidth();
	}

	void HandleSelection()
	{
		m_select_links.clear();
		m_select_nodes.clear();

		if (ImNodes::NumSelectedLinks() > 0)
		{
			m_select_links.resize(ImNodes::NumSelectedLinks());
			ImNodes::GetSelectedLinks(m_select_links.data());
		}

		if (ImNodes::NumSelectedNodes() > 0)
		{
			m_select_nodes.resize(ImNodes::NumSelectedNodes());
			ImNodes::GetSelectedNodes(m_select_nodes.data());
		}
	}

	void DrawEdges(Resource<ResourceType::Material> *resource)
	{
		for (auto &[dst, src] : resource->GetDesc().GetEdges())
		{
			ImNodes::Link(static_cast<int32_t>(Hash(src, dst)), static_cast<int32_t>(src), static_cast<int32_t>(dst));
		}
	}

	void AddEdge(Resource<ResourceType::Material> *resource)
	{
		int32_t src = 0, dst = 0;
		if (ImNodes::IsLinkCreated(&src, &dst))
		{
			resource->GetDesc().Link(src, dst);
		}
	}

  private:
	ImNodesEditorContext *m_context = nullptr;

	std::string m_material_name = "";

	size_t m_current_handle = 0;

	std::vector<int32_t> m_select_nodes;
	std::vector<int32_t> m_select_links;

	std::map<MaterialNodePin::Type, uint32_t> m_pin_color = {
	    {MaterialNodePin::Type::BSDF, IM_COL32(0, 128, 0, 255)},
	    {MaterialNodePin::Type::Media, IM_COL32(0, 0, 128, 255)},
	    {MaterialNodePin::Type::Float, IM_COL32(0, 128, 128, 255)},
	    {MaterialNodePin::Type::Float3, IM_COL32(128, 128, 0, 255)},
	    {MaterialNodePin::Type::RGB, IM_COL32(128, 128, 0, 255)},
	};
};

extern "C"
{
	EXPORT_API MaterialGraphEditor *Create(Editor *editor, ImGuiContext *context)
	{
		ImGui::SetCurrentContext(context);
		return new MaterialGraphEditor(editor);
	}
}