#include "RenderGraphEditor.hpp"

#include <RenderCore/RenderGraph/RenderGraph.hpp>

#include <Renderer/BasePass/VisibilityBufferPass.hpp>

#include <imnodes.h>

namespace Ilum
{
RenderGraphEditor::RenderGraphEditor(Editor *editor) :
    Widget("Render Graph Editor", editor)
{
}

RenderGraphEditor::~RenderGraphEditor()
{
}

void RenderGraphEditor::Tick()
{
	ImGui::Begin(m_name.c_str(), &m_active, ImGuiWindowFlags_MenuBar);

	if (ImGui::BeginMenuBar())
	{
		if (ImGui::MenuItem("Load"))
		{
		}
		if (ImGui::BeginMenu("Add"))
		{
			if (ImGui::BeginMenu("Pass"))
			{
				auto *pass_name = RenderPassNameList;
				while (pass_name)
				{
					if (ImGui::MenuItem(pass_name->name))
					{
						RGHandle       handle(m_current_handle++);
						RenderPassDesc desc = rttr::type::invoke(fmt::format("{}_Desc", pass_name->name).c_str(), {m_current_handle}).get_value<RenderPassDesc>();
						m_desc.passes.emplace(handle, desc);
					}
					pass_name = pass_name->next;
				}
				ImGui::EndMenu();
			}
			if (ImGui::MenuItem("Texture"))
			{
			}
			if (ImGui::MenuItem("Buffer"))
			{
			}
			ImGui::EndMenu();
		}
		if (ImGui::MenuItem("Compile"))
		{
		}
		ImGui::EndMenuBar();
	}

	ImNodes::BeginNodeEditor();

	// Draw pass nodes
	{
		for (auto &[handle, pass] : m_desc.passes)
		{
			const float node_width = 200.0f;

			ImNodes::BeginNode(static_cast<int32_t>(handle.GetHandle()));
			ImNodes::BeginNodeTitleBar();
			ImGui::Text(pass.name.c_str());
			ImNodes::EndNodeTitleBar();

			// Draw read pin
			for (auto &[name, read] : pass.reads)
			{
				if (read.first == RenderPassDesc::ResourceType::Texture)
				{
					ImNodes::BeginInputAttribute(static_cast<int32_t>(read.second.GetHandle()), ImNodesPinShape_CircleFilled);
					ImGui::TextUnformatted(name.c_str());
					ImNodes::EndInputAttribute();
				}
				else
				{
					ImNodes::BeginInputAttribute(static_cast<int32_t>(read.second.GetHandle()), ImNodesPinShape_TriangleFilled);
					ImGui::TextUnformatted(name.c_str());
					ImNodes::EndInputAttribute();
				}
			}

			// Draw write pin
			for (auto &[name, write] : pass.writes)
			{
				if (write.first == RenderPassDesc::ResourceType::Texture)
				{
					ImNodes::BeginOutputAttribute(static_cast<int32_t>(write.second.GetHandle()), ImNodesPinShape_CircleFilled);
					const float label_width = ImGui::CalcTextSize(name.c_str()).x;
					ImGui::Indent(node_width - label_width);
					ImGui::TextUnformatted(name.c_str());
					ImNodes::EndOutputAttribute();
				}
				else
				{
					ImNodes::BeginOutputAttribute(static_cast<int32_t>(write.second.GetHandle()), ImNodesPinShape_TriangleFilled);
					const float label_width = ImGui::CalcTextSize(name.c_str()).x;
					ImGui::Indent(node_width - label_width);
					ImGui::TextUnformatted(name.c_str());
					ImNodes::EndOutputAttribute();
				}
			}
			ImNodes::EndNode();
		}
	}

	// Draw Texture Node
	{
		for (auto &[handle, texture] : m_desc.textures)
		{
			ImNodes::PushColorStyle(ImNodesCol_TitleBar, IM_COL32(255, 0, 0, 255));
			ImNodes::BeginNode(static_cast<int32_t>(handle.GetHandle()));
			ImNodes::BeginNodeTitleBar();
			ImNodes::EndNodeTitleBar();

			ImNodes::EndNode();
			ImNodes::PopColorStyle();
		}
	}

	// Draw Buffer Node
	{

	}

	// Draw Edges
	{

	}

	// Create New Edges
	{

	}

	ImNodes::EndNodeEditor();

	ImGui::End();
}
}        // namespace Ilum