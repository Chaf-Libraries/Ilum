#pragma once

#include "MaterialGraphEditor.hpp"
#include "ImGui/ImGuiHelper.hpp"

#include <Core/Path.hpp>

#include <imnodes.h>

#pragma warning(push, 0)
#include <nfd.h>
#pragma warning(pop)

namespace Ilum
{
MaterialGraphEditor::MaterialGraphEditor(Editor *editor) :
    Widget("Material Graph Editor", editor)
{
	m_context = ImNodes::EditorContextCreate();
}

MaterialGraphEditor::~MaterialGraphEditor()
{
	ImNodes::EditorContextFree(m_context);
}

void MaterialGraphEditor::Tick()
{
	ImGui::Begin(m_name.c_str(), &m_active, ImGuiWindowFlags_MenuBar);

	ImNodes::EditorContextSet(m_context);

	DrawMenu();

	// Handle selection
	std::vector<int32_t> selected_links;
	std::vector<int32_t> selected_nodes;

	if (ImNodes::NumSelectedLinks() > 0)
	{
		selected_links.resize(ImNodes::NumSelectedLinks());
		ImNodes::GetSelectedLinks(selected_links.data());
	}

	if (ImNodes::NumSelectedNodes() > 0)
	{
		selected_nodes.resize(ImNodes::NumSelectedNodes());
		ImNodes::GetSelectedNodes(selected_nodes.data());
	}

	ImGui::Columns(2);
	ImGui::SetColumnWidth(0, ImGui::GetWindowWidth() * 0.8f);

	ImNodes::BeginNodeEditor();

	// Popup Window
	if (!selected_links.empty() || !selected_nodes.empty())
	{
		if (ImGui::BeginPopupContextWindow(0, 1, true))
		{
			if (ImGui::MenuItem("Remove"))
			{
				for (const auto &node : selected_nodes)
				{
					m_desc.EraseNode(static_cast<size_t>(node));
				}
				for (const auto &link : selected_links)
				{
					for (auto iter = m_desc.links.begin(); iter != m_desc.links.end();)
					{
						if (static_cast<int32_t>(Hash(iter->second, iter->first)) == link)
						{
							iter = m_desc.links.erase(iter);
						}
						else
						{
							iter++;
						}
					}
				}
			}
			ImGui::EndPopup();
		}
	}

	// Draw nodes
	for (auto &[node_handle, desc] : m_desc.nodes)
	{
		const float node_width = 120.0f;

		ImGui::PushItemWidth(100.f);
		ImNodes::BeginNode(static_cast<int32_t>(node_handle));
		ImNodes::BeginNodeTitleBar();
		ImGui::Text(desc.name.c_str());
		ImNodes::EndNodeTitleBar();
		ImNodes::BeginStaticAttribute(static_cast<int32_t>(node_handle));
		ImGui::EditVariant("", desc.data);
		ImNodes::EndStaticAttribute();
		// Draw Pin
		for (auto &[pin_handle, pin] : desc.pins)
		{
			if (pin.attribute == MaterialNodePin::Attribute::Input)
			{
				ImNodes::BeginInputAttribute(static_cast<int32_t>(pin.handle));
				ImGui::Text(pin.name.c_str());
				ImGui::SameLine();
				if (!m_desc.HasLink(pin.handle))
				{
					ImGui::EditVariant("", pin.data);
				}
				ImNodes::EndInputAttribute();
			}
			if (pin.attribute == MaterialNodePin::Attribute::Output)
			{
				ImNodes::BeginOutputAttribute(static_cast<int32_t>(pin.handle));
				const float label_width = ImGui::CalcTextSize(pin.name.c_str()).x;
				ImGui::Indent(node_width - label_width);
				ImGui::Text(pin.name.c_str());
				ImGui::SameLine();
				ImNodes::EndOutputAttribute();
			}
		}
		ImGui::PopItemWidth();
		ImNodes::EndNode();
	}

	// Draw Edges
	{
		for (auto &[target, source] : m_desc.links)
		{
			// ImNodes::PushColorStyle(ImNodesCol_Link, color);
			ImNodes::Link(static_cast<int32_t>(Hash(source, target)), static_cast<int32_t>(source), static_cast<int32_t>(target));
			// ImNodes::PopColorStyle();
		}
	}

	ImNodes::EndNodeEditor();

	// Create New Edges
	{
		int32_t src = 0, dst = 0;
		if (ImNodes::IsLinkCreated(&src, &dst))
		{
			m_desc.Link(static_cast<size_t>(src), static_cast<size_t>(dst));
		}
	}

	// Inspector
	{
		ImGui::BeginChild("Render Graph Inspector", ImVec2(0, 0), false, ImGuiWindowFlags_HorizontalScrollbar);

		ImGui::PushItemWidth(ImGui::GetColumnWidth(1) * 0.7f);

		ImGui::PopItemWidth();

		if (ImGui::GetScrollY() >= ImGui::GetScrollMaxY())
		{
			ImGui::SetScrollHereY(1.0f);
		}

		ImGui::EndChild();
	}

	ImGui::End();
}

void MaterialGraphEditor::DrawMenu()
{
	if (ImGui::BeginMenuBar())
	{
		if (ImGui::BeginMenu("File"))
		{
			if (ImGui::MenuItem("Load"))
			{
				char *path = nullptr;
				if (NFD_OpenDialog("mat", Path::GetInstance().GetCurrent(false).c_str(), &path) == NFD_OKAY)
				{
					std::string editor_state = "";
					DESERIALIZE(path, m_desc, editor_state);
					ImNodes::LoadEditorStateFromIniString(m_context, editor_state.data(), editor_state.size());

					// Update max current id
					m_current_handle = 0;

					for (auto &[node_handle, node] : m_desc.nodes)
					{
						m_current_handle = std::max(m_current_handle, node_handle);
						for (auto &[pin_name, pin] : node.pins)
						{
							m_current_handle = std::max(m_current_handle, pin.handle);
						}
					}

					free(path);
				}
			}

			if (ImGui::MenuItem("Save"))
			{
				char *path = nullptr;
				if (NFD_SaveDialog("mat", Path::GetInstance().GetCurrent(false).c_str(), &path) == NFD_OKAY)
				{
					std::string dir      = Path::GetInstance().GetFileDirectory(path);
					std::string filename = Path::GetInstance().GetFileName(path, false);
					SERIALIZE(dir + filename + ".mat", m_desc, std::string(ImNodes::SaveCurrentEditorStateToIniString()));
					// p_editor->GetRenderer()->GetResourceManager()->Import<ResourceType::RenderGraph>(dir + filename + ".mat");
					free(path);
				}
			}

			ImGui::EndMenu();
		}

		if (ImGui::BeginMenu("Add"))
		{
			for (const auto &type : rttr::type::get_types())
			{
				if (type.get_metadata("MaterialNode").is_valid())
				{
					bool has_create   = false;
					bool has_category = type.get_metadata("Category").is_valid();

					if (has_category ? ImGui::BeginMenu(type.get_metadata("Category").get_value<std::string>().c_str()) : true)
					{
						if (ImGui::MenuItem(type.get_metadata("MaterialNode").get_value<std::string>().c_str()))
						{
							auto var       = type.create();
							auto node_desc = rttr::type::get(var).get_method("Create").invoke(var, m_current_handle).convert<MaterialNodeDesc>();
							m_desc.AddNode(m_current_handle, std::move(node_desc));
							has_create = true;
						}

						if (has_category)
						{
							ImGui::EndMenu();
						}

						if (has_create)
						{
							break;
						}
					}
				}
			}

			ImGui::EndMenu();
		}

		if (ImGui::MenuItem("Clear"))
		{
			m_desc.links.clear();
			m_desc.nodes.clear();
			m_desc.node_query.clear();
		}

		ImGui::EndMenuBar();
	}
}
}        // namespace Ilum