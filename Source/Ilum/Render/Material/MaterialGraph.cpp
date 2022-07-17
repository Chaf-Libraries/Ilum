#include "MaterialGraph.hpp"
#include "MaterialNode/Constant.hpp"
#include "MaterialNode/MaterialNode.hpp"
#include "MaterialNode/Operator.hpp"

#include <RHI/ImGuiContext.hpp>

namespace Ilum
{
MaterialGraph::MaterialGraph(AssetManager *asset_manager, const std::string &name) :
    m_asset_manager(asset_manager),
    m_name(name)
{
	AddPin(0, PinType::Float);
}

MaterialGraph::~MaterialGraph()
{
	m_node_lookup.clear();
	m_edges.clear();
	m_pin_callbacks.clear();
	m_pin_type.clear();
	m_nodes.clear();
}

const std::string MaterialGraph::CompileToHLSL()
{
	std::string result = "";
	for (auto &[pin, callback] : m_pin_callbacks)
	{
		result += callback();
	}
	return result;
}

void MaterialGraph::EnableEditor()
{
	m_enable_editor = true;
}

void MaterialGraph::OnImGui(ImGuiContext &context)
{
	static auto editor_context = ImNodes::EditorContextCreate();

	ImGui::Begin("Material Editor", &m_enable_editor);
	ImNodes::EditorContextSet(editor_context);

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

	for (auto &id : selected_nodes)
	{
		if (m_node_lookup.find(id) != m_node_lookup.end())
		{
			ImGui::PushID(id);
			if (ImGui::TreeNode(m_node_lookup.at(id)->GetName().c_str()))
			{
				m_node_lookup.at(id)->OnImGui(context);
				ImGui::TreePop();
			}
			ImGui::PopID();
		}
	}

	ImGui::NextColumn();

	ImNodes::BeginNodeEditor();

	// Popup Window
	if (ImGui::BeginPopupContextWindow(0, 1, true))
	{
		// Compile
		if (ImGui::MenuItem("Compile"))
		{
			std::string result  = "";
			size_t out_pin = 0;
			if (LinkTo(out_pin, 0) && m_pin_callbacks.find(out_pin) != m_pin_callbacks.end())
			{
				result = m_pin_callbacks.at(out_pin)();
			}
		}

		// Remove
		if (!selected_links.empty() || !selected_nodes.empty())
		{
			if (ImGui::MenuItem("Remove"))
			{
				for (auto &id : selected_links)
				{
					m_edges.erase(static_cast<size_t>(id));
				}
				for (auto &id : selected_nodes)
				{
					EraseNode(id);
				}
			}
		}

		// New Node
		if (ImGui::BeginMenu("New Node"))
		{
			// Constant
			if (ImGui::BeginMenu("Constant"))
			{
				for (auto &[dim, type_create] : MGNode::ConstantNodeCreation)
				{
					if (ImGui::BeginMenu(dim))
					{
						for (auto &[type, create] : type_create)
						{
							if (ImGui::MenuItem(type))
							{
								AddNode(std::move(create(this)));
							}
						}
						ImGui::EndMenu();
					}
				}
				ImGui::EndMenu();
			}

			// Operator
			if (ImGui::BeginMenu("Operator"))
			{
				for (auto &[type, create] : MGNode::OperatorNodeCreation)
				{
					if (ImGui::MenuItem(type))
					{
						AddNode(std::move(create(this)));
					}
				}
				ImGui::EndMenu();
			}
			ImGui::EndMenu();
		}

		ImGui::EndPopup();
	}

	// Draw Nodes
	for (auto &node : m_nodes)
	{
		node->OnImnode();
	}

	// Draw Output Node
	{
		ImNodes::BeginNode(0);

		ImNodes::BeginNodeTitleBar();
		ImGui::Text("Output");
		ImNodes::EndNodeTitleBar();

		ImNodes::BeginInputAttribute(0);
		ImGui::Text("BxDF");
		ImNodes::EndInputAttribute();

		ImNodes::EndNode();
	}

	// Draw Links
	{
		for (auto &[id, edge] : m_edges)
		{
			ImNodes::Link(static_cast<int32_t>(id), static_cast<int32_t>(edge.first), static_cast<int32_t>(edge.second));
		}
	}

	ImNodes::MiniMap();
	ImNodes::EndNodeEditor();

	// Create Link
	{
		int32_t from = 0, to = 0;
		if (ImNodes::IsLinkCreated(&from, &to))
		{
			Link(from, to);
		}
	}

	ImGui::End();
}

size_t MaterialGraph::NewNodeID()
{
	return m_node_id++;
}

size_t MaterialGraph::NewPinID()
{
	return m_pin_id++;
}

void MaterialGraph::Link(size_t from, size_t to)
{
	if (m_pin_type.find(from) != m_pin_type.end() &&
	    m_pin_type.find(to) != m_pin_type.end() &&
	    m_pin_type[from] == m_pin_type[to])
	{
		for (auto &[id, edge] : m_edges)
		{
			if (edge.second == to)
			{
				m_edges.erase(id);
				break;
			}
		}
		m_edges.emplace(m_edge_id++, std::make_pair(from, to));
	}
}

void MaterialGraph::UnLink(size_t from, size_t to)
{
	for (auto iter = m_edges.begin(); iter != m_edges.end(); iter++)
	{
		if (iter->second.first == from && iter->second.second == to)
		{
			iter = m_edges.erase(iter);
			return;
		}
	}
}

bool MaterialGraph::LinkTo(size_t &from, size_t to)
{
	for (auto &[id, edge] : m_edges)
	{
		if (edge.second == to)
		{
			from = edge.first;
			return true;
		}
	}
	return false;
}

std::string MaterialGraph::CallPin(size_t pin)
{
	if (m_pin_callbacks.find(pin) != m_pin_callbacks.end())
	{
		return m_pin_callbacks.at(pin)();
	}

	return "";
}

void MaterialGraph::BindPinCallback(size_t pin, std::function<std::string(void)> &&callback)
{
	m_pin_callbacks.emplace(pin, std::move(callback));
}

void MaterialGraph::UnbindPinCallback(size_t pin)
{
	if (m_pin_callbacks.find(pin) != m_pin_callbacks.end())
	{
		m_pin_callbacks.erase(pin);
	}
}

void MaterialGraph::AddPin(size_t pin, PinType type)
{
	m_pin_type.emplace(pin, type);
}

void MaterialGraph::SetPin(size_t pin, PinType type)
{
	if (m_pin_type.find(pin) != m_pin_type.end())
	{
		m_pin_type[pin] = type;
	}
}

void MaterialGraph::ErasePin(size_t pin)
{
	if (m_pin_type.find(pin) != m_pin_type.end())
	{
		m_pin_type.erase(pin);
		for (auto iter = m_edges.begin(); iter != m_edges.end(); iter++)
		{
			if (iter->second.first == pin ||
			    iter->second.second == pin)
			{
				iter = m_edges.erase(iter);
			}
		}
	}
}

void MaterialGraph::AddNode(std::unique_ptr<MaterialNode> &&node)
{
	auto &node_ptr = m_nodes.emplace_back(std::move(node));
	m_node_lookup.emplace(node_ptr->GetNodeID(), node_ptr.get());
}

void MaterialGraph::EraseNode(size_t node)
{
	m_node_lookup.erase(node);
	for (auto iter = m_nodes.begin(); iter != m_nodes.end(); iter++)
	{
		if ((*iter)->GetNodeID() == node)
		{
			m_nodes.erase(iter);
			return;
		}
	}
}

AssetManager *MaterialGraph::GetAssetManager() const
{
	return m_asset_manager;
}

}        // namespace Ilum