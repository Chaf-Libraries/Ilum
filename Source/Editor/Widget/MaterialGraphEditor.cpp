#pragma once

#include "MaterialGraphEditor.hpp"
#include "Editor.hpp"
#include "ImGui/ImGuiHelper.hpp"

#include <Core/Path.hpp>
#include <RHI/RHIContext.hpp>
#include <MaterialGraph/MaterialGraphBuilder.hpp>
#include <Renderer/Renderer.hpp>
#include <Resource/ResourceManager.hpp>

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
	auto *material_graph = p_editor->GetRenderer()->GetResourceManager()->GetResource<ResourceType::Material>(m_uuid) ?
	                           p_editor->GetRenderer()->GetResourceManager()->GetResource<ResourceType::Material>(m_uuid)->Get() :
                               nullptr;

	if (material_graph)
	{
		material_graph->SetUpdate(false);
	}

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

	ImNodes::BeginNodeEditor();

	if (material_graph)
	{
		// Popup Window
		if (!selected_links.empty() || !selected_nodes.empty())
		{
			if (ImGui::BeginPopupContextWindow(0, 1, true))
			{
				if (ImGui::MenuItem("Remove"))
				{
					for (const auto &node : selected_nodes)
					{
						material_graph->GetDesc().EraseNode(static_cast<size_t>(node));
					}
					for (const auto &link : selected_links)
					{
						for (auto iter = material_graph->GetDesc().links.begin(); iter != material_graph->GetDesc().links.end();)
						{
							if (static_cast<int32_t>(Hash(iter->second, iter->first)) == link)
							{
								iter     = material_graph->GetDesc().links.erase(iter);
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
		for (auto &[node_handle, desc] : material_graph->GetDesc().nodes)
		{
			float node_width = 100.f;

			bool update = false;

			ImGui::PushItemWidth(100.f);
			ImNodes::BeginNode(static_cast<int32_t>(node_handle));
			ImNodes::BeginNodeTitleBar();
			ImGui::Text(desc.name.c_str());
			node_width = std::max(node_width, ImGui::CalcTextSize(desc.name.c_str()).x);
			ImNodes::EndNodeTitleBar();
			ImNodes::BeginStaticAttribute(static_cast<int32_t>(node_handle));
			auto type = desc.data.get_type();
			if (desc.data)
			{
				update |= ImGui::EditVariant("", p_editor, desc.data);
			}
			ImNodes::EndStaticAttribute();
			// Draw Pin
			for (auto &[pin_name, pin] : desc.pins)
			{
				if (!pin.enable)
				{
					continue;
				}

				if (pin.attribute == MaterialNodePin::Attribute::Input)
				{
					ImNodes::BeginInputAttribute(static_cast<int32_t>(pin.handle));
					ImGui::Text(pin.name.c_str());
					float item_width = ImGui::CalcTextSize(pin.name.c_str()).x;
					if (!material_graph->GetDesc().HasLink(pin.handle))
					{
						ImGui::SameLine();
						update |= ImGui::EditVariant("", p_editor, pin.data);
						item_width += ImGui::CalcItemWidth();
					}
					node_width = std::max(node_width, item_width);
					ImNodes::EndInputAttribute();
				}
				if (pin.attribute == MaterialNodePin::Attribute::Output)
				{
					ImNodes::BeginOutputAttribute(static_cast<int32_t>(pin.handle));
					if (pin.data)
					{
						update |= ImGui::EditVariant("", p_editor, pin.data);
						ImGui::SameLine();
					}
					const float label_width = ImGui::CalcTextSize(pin.name.c_str()).x;
					ImGui::Indent(node_width - label_width);
					ImGui::Text(pin.name.c_str());
					ImNodes::EndOutputAttribute();
				}
			}
			ImGui::PopItemWidth();
			ImNodes::EndNode();

			if (update)
			{
				material_graph->GetDesc().UpdateNode(node_handle);
			}
		}

		// Draw Edges
		{
			for (auto &[target, source] : material_graph->GetDesc().links)
			{
				// ImNodes::PushColorStyle(ImNodesCol_Link, color);
				ImNodes::Link(static_cast<int32_t>(Hash(source, target)), static_cast<int32_t>(source), static_cast<int32_t>(target));
				// ImNodes::PopColorStyle();
			}
		}
	}

	ImNodes::MiniMap(0.1f);
	ImNodes::EndNodeEditor();

	// Drag Drop Material
	{
		if (ImGui::BeginDragDropTarget())
		{
			if (const auto *pay_load = ImGui::AcceptDragDropPayload(rttr::type::get<ResourceType>().get_enumeration().value_to_name(ResourceType::Material).to_string().c_str()))
			{
				ASSERT(pay_load->DataSize == sizeof(size_t));
				size_t uuid     = *static_cast<size_t *>(pay_load->Data);
				auto  *resource = p_editor->GetRenderer()->GetResourceManager()->GetResource<ResourceType::Material>(uuid);
				if (resource)
				{
					m_uuid = uuid;

					std::string editor_state = resource->GetEditorState();
					ImNodes::LoadEditorStateFromIniString(m_context, editor_state.data(), editor_state.length());

					for (auto &[node_handle, node] : resource->Get()->GetDesc().nodes)
					{
						m_current_handle = std::max(node_handle, m_current_handle);
						for (auto &[pin_handle, pin] : node.pins)
						{
							m_current_handle = std::max(pin_handle, m_current_handle);
						}
					}
				}
			}
		}
	}

	// Create New Edges
	{
		int32_t src = 0, dst = 0;
		if (ImNodes::IsLinkCreated(&src, &dst))
		{
			material_graph->GetDesc().Link(static_cast<size_t>(src), static_cast<size_t>(dst));
		}
	}

	ImGui::End();
}

void MaterialGraphEditor::DrawMenu()
{
	auto *material_graph = p_editor->GetRenderer()->GetResourceManager()->GetResource<ResourceType::Material>(m_uuid) ?
	                           p_editor->GetRenderer()->GetResourceManager()->GetResource<ResourceType::Material>(m_uuid)->Get() :
                               nullptr;

	if (ImGui::BeginMenuBar())
	{
		if (ImGui::MenuItem("New"))
		{
			ImGui::OpenPopup("New Material");
		}
		if (ImGui::BeginPopup("New Material"))
		{
			static std::string name;
			ImGui::EditVariant("", p_editor, name);
			ImGui::SameLine();
			if (ImGui::Button("Create"))
			{
				m_uuid = Hash(name);
				p_editor->GetRenderer()->GetResourceManager()->CreateResource<ResourceType::Material>(m_uuid);
				MaterialGraphDesc desc;
				desc.name        = name;
				std::string meta = fmt::format("Name: {}", name);
				SERIALIZE("Asset/Meta/" + std::to_string(m_uuid) + ".asset", ResourceType::Material, m_uuid, meta, desc, std::string(ImNodes::SaveCurrentEditorStateToIniString()));
			}
			ImGui::EndPopup();
		}

		if (material_graph)
		{
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
								material_graph->GetDesc().AddNode(m_current_handle, std::move(node_desc));
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
				material_graph->GetDesc().links.clear();
				material_graph->GetDesc().nodes.clear();
				material_graph->GetDesc().node_query.clear();
			}

			if (ImGui::MenuItem("Compile"))
			{
				MaterialGraphBuilder builder(p_editor->GetRHIContext());
				builder.Compile(material_graph);
				p_editor->GetRenderer()->GetResourceManager()->GetResource<ResourceType::Material>(m_uuid)->Save(ImNodes::SaveCurrentEditorStateToIniString());
				p_editor->GetRenderer()->Reset();
				material_graph->SetUpdate(true);
			}

			if (material_graph)
			{
				if (ImGui::Button(material_graph->GetDesc().name.c_str()))
				{
					m_uuid = size_t(~0);
				}
			}
		}

		ImGui::EndMenuBar();
	}
}
}        // namespace Ilum