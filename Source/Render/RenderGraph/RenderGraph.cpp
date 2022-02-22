#include "RenderGraph.hpp"

#include <imnodes.h>

namespace Ilum::Render
{
void RenderGraph::RegisterPin(int32_t pin, int32_t node)
{
	m_pin_query[pin] = node;
}

void RenderGraph::UnRegisterPin(int32_t pin)
{
	if (m_pin_query.find(pin) != m_pin_query.end())
	{
		m_pin_query.erase(pin);
	}
	for (auto iter = m_links.begin(); iter != m_links.end();)
	{
		if (iter->first == pin || iter->second == pin)
		{
			iter = m_links.erase(iter);
		}
		else
		{
			iter++;
		}
	}
}

void RenderGraph::OnImGui()
{
	ImNodes::BeginNodeEditor();

	for (auto &[uuid, node] : m_resource_nodes)
	{
		node->OnImNode();
	}

	for (auto &[uuid, node] : m_pass_nodes)
	{
		node->OnImNode();
	}

	// Link
	for (int i = 0; i < m_links.size(); ++i)
	{
		const std::pair<int, int> p = m_links[i];
		ImNodes::Link(i, p.first, p.second);
	}

	ImNodes::MiniMap();
	ImNodes::EndNodeEditor();

	//int link;
	//if (ImNodes::IsLinkHovered(&link))
	//{
	//	if (ImGui::IsMouseClicked(ImGuiMouseButton_Right))
	//	{
	//		m_links.erase(m_links.begin() + link);
	//	}
	//}

	int start_attr, end_attr;
	if (ImNodes::IsLinkCreated(&start_attr, &end_attr))
	{
		// resource - > pass
		if (m_resource_nodes.find(m_pin_query[start_attr]) != m_resource_nodes.end() &&
		    m_pass_nodes.find(m_pin_query[end_attr]) != m_pass_nodes.end())
		{
			if (m_resource_nodes[m_pin_query[start_attr]]->ReadBy(m_pass_nodes[m_pin_query[end_attr]].get(), end_attr))
			{
				m_links.push_back(std::make_pair(start_attr, end_attr));
			}
		}

		// pass - > resource
		if (m_pass_nodes.find(m_pin_query[start_attr]) != m_pass_nodes.end() &&
		    m_resource_nodes.find(m_pin_query[end_attr]) != m_resource_nodes.end())
		{
			if (m_resource_nodes[m_pin_query[start_attr]]->WriteBy(m_pass_nodes[m_pin_query[end_attr]].get(), end_attr))
			{
				m_links.push_back(std::make_pair(start_attr, end_attr));
			}
		}
	}

	ImGui::Begin("Render Node Inspector");
	if (ImNodes::NumSelectedNodes() > 0)
	{
		std::vector<int32_t> select_nodes(ImNodes::NumSelectedNodes());
		ImNodes::GetSelectedNodes(select_nodes.data());

		const ImGuiTreeNodeFlags tree_node_flags = ImGuiTreeNodeFlags_DefaultOpen | ImGuiTreeNodeFlags_Framed | ImGuiTreeNodeFlags_SpanAvailWidth | ImGuiTreeNodeFlags_AllowItemOverlap | ImGuiTreeNodeFlags_FramePadding;
		for (auto &node : select_nodes)
		{
			if (ImGui::TreeNodeEx(std::to_string(m_render_nodes[node]->GetUUID()).c_str(), tree_node_flags, m_render_nodes[node]->GetName().c_str()))
			{
				m_render_nodes[node]->OnImGui();
				ImGui::TreePop();
			}
		}
	}
	ImGui::End();
}
}        // namespace Ilum::Render