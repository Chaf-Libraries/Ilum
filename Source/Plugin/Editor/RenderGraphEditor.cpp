#include <Editor/Editor.hpp>
#include <Editor/Widget.hpp>
#include <RenderGraph/RenderGraph.hpp>
#include <RenderGraph/RenderGraphBuilder.hpp>
#include <RenderGraph/RenderPass.hpp>
#include <Renderer/Renderer.hpp>
#include <Resource/Resource/RenderPipeline.hpp>
#include <Resource/ResourceManager.hpp>

#include <imgui.h>
#include <imgui_internal.h>
#include <imnodes/imnodes.h>

#include <nfd.h>

using namespace Ilum;

class RenderGraphEditor : public Widget
{
  public:
	RenderGraphEditor(Editor *editor) :
	    Widget("Render Graph Editor", editor)
	{
		ImNodes::CreateContext();
		m_context = ImNodes::EditorContextCreate();
	}

	virtual ~RenderGraphEditor() override
	{
		ImNodes::DestroyContext();
		ImNodes::EditorContextFree(m_context);
	}

	virtual void Tick() override
	{
		if (!ImGui::Begin(m_name.c_str(), nullptr))
		{
			ImGui::End();
			return;
		}

		auto *resource_manager = p_editor->GetRenderer()->GetResourceManager();
		auto *resource = resource_manager->Get<ResourceType::RenderPipeline>(m_pipeline_name);

		ImGui::Columns(2);

		{
			ImGui::BeginChild("Render Pipeline Editor Inspector", ImVec2(0, 0), false, ImGuiWindowFlags_HorizontalScrollbar);

			ImGui::Text("Render Pipeline Editor Inspector");

			SetRenderPipeline(resource, resource_manager);

			if (resource)
			{
				for (auto &node : m_select_nodes)
				{
					if (resource->GetDesc().HasPass(node))
					{
						auto &pass = resource->GetDesc().GetPass(static_cast<size_t>(node));
						if (!pass.GetConfig().Empty())
						{
							ImGui::Separator();
							PluginManager::GetInstance().Call<bool>(fmt::format("shared/RenderPass/RenderPass.{}.{}.dll", pass.GetCategory(), pass.GetName()), "OnImGui", &pass.GetConfig(), ImGui::GetCurrentContext());
						}
					}
				}
			}

			if (ImGui::GetScrollY() >= ImGui::GetScrollMaxY())
			{
				ImGui::SetScrollHereY(1.0f);
			}

			ImGui::EndChild();
		}

		ImGui::NextColumn();

		{
			ImGui::BeginChild("Render Graph Editor", ImVec2(0, 0), false, ImGuiWindowFlags_MenuBar);

			if (resource)
			{
				MainMenuBar(resource);
			}

			HandleSelection();

			ImNodes::BeginNodeEditor();

			if (resource)
			{
				for (auto &new_node : m_new_nodes)
				{
					ImNodes::SetNodeScreenSpacePos(new_node, ImVec2(ImGui::GetContentRegionAvail().x, ImGui::GetContentRegionAvail().y));
				}
				m_new_nodes.clear();

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
		}

		ImGui::End();
	}

  private:
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

	void AddEdge(Resource<ResourceType::RenderPipeline> *resource)
	{
		int32_t src = 0, dst = 0;
		if (ImNodes::IsLinkCreated(&src, &dst))
		{
			resource->GetDesc().Link(static_cast<size_t>(glm::abs(src)), static_cast<size_t>(glm::abs(dst)));
		}
	}

	void PopupWindow(Resource<ResourceType::RenderPipeline> *resource)
	{
		if (ImGui::BeginPopupContextWindow(0, ImGuiPopupFlags_MouseButtonRight))
		{
			if (!m_select_links.empty() || !m_select_nodes.empty())
			{
				if (ImGui::MenuItem("Remove"))
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
						resource->GetDesc().ErasePass(node);
					}
				}

				m_select_nodes.clear();
				m_select_links.clear();
			}

			ImGui::EndPopup();
		}
	}

	void MainMenuBar(Resource<ResourceType::RenderPipeline> *resource)
	{
		if (ImGui::BeginMenuBar())
		{
			if (ImGui::BeginMenu("Pass"))
			{
				for (const auto &file : std::filesystem::directory_iterator("shared/RenderPass/"))
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
								RenderPassDesc desc;
								PluginManager::GetInstance().Call(file.path().string(), "Create", &desc, &m_current_handle);
								m_new_nodes.push_back(static_cast<int32_t>(m_current_handle));
								resource->GetDesc().AddPass(m_current_handle++, std::move(desc));
							}
							ImGui::EndMenu();
						}
					}
				}
				ImGui::EndMenu();
			}

			if (ImGui::MenuItem("Compile"))
			{
				auto *rhi_context  = p_editor->GetRHIContext();
				auto *renderer     = p_editor->GetRenderer();
				auto  render_graph = resource->Compile(rhi_context, renderer, renderer->GetViewport(), ImNodes::SaveCurrentEditorStateToIniString());

				if (render_graph)
				{
					renderer->SetRenderGraph(std::move(render_graph));
				}
				else
				{
					LOG_INFO("Render Graph Compile Failed!");
				}
			}

			if (!resource->GetDesc().GetPasses().empty())
			{
				if (ImGui::MenuItem("Clear"))
				{
					resource->GetDesc().Clear();
				}
			}

			ImGui::EndMenuBar();
		}
	}

	void DrawNodes(Resource<ResourceType::RenderPipeline> *resource)
	{
		for (auto &[node_handle, node_desc] : resource->GetDesc().GetPasses())
		{
			const float node_width = glm::max(ImGui::CalcTextSize(node_desc.GetName().c_str()).x, 120.f);

			ImNodes::BeginNode(static_cast<int32_t>(node_handle));
			ImNodes::BeginNodeTitleBar();
			ImGui::Text(node_desc.GetName().c_str());
			ImNodes::EndNodeTitleBar();

			// Draw exec pin
			{
				ImNodes::PushColorStyle(ImNodesCol_Pin, IM_COL32(255, 255, 255, 255));
				ImNodes::PushColorStyle(ImNodesCol_PinHovered, IM_COL32(255, 255, 255, 255));
				ImNodes::PushColorStyle(ImNodesCol_Link, IM_COL32(255, 255, 255, 255));

				// In attribute
				ImNodes::BeginInputAttribute(static_cast<int32_t>(node_handle));
				ImGui::TextUnformatted("In");
				ImNodes::EndInputAttribute();

				ImGui::SameLine();

				// Out attribute
				ImNodes::BeginOutputAttribute(-static_cast<int32_t>(node_handle));
				const float label_width = ImGui::CalcTextSize("OutIn ").x;
				ImGui::Indent(node_width - label_width);
				ImGui::TextUnformatted("Out");
				ImNodes::EndOutputAttribute();

				ImNodes::PopColorStyle();
				ImNodes::PopColorStyle();
				ImNodes::PopColorStyle();
			}

			for (auto &[pin_handle, pin] : node_desc.GetPins())
			{
				if (pin.attribute == RenderPassPin::Attribute::Input)
				{
					ImNodes::PushColorStyle(ImNodesCol_Pin, m_color[pin.type]);
					ImNodes::BeginInputAttribute(static_cast<int32_t>(pin_handle));
					ImGui::TextUnformatted(pin.name.c_str());
					if (!resource->GetDesc().HasLink(pin_handle))
					{
						ImGui::SameLine();
						ImGui::PushItemWidth(node_width - ImGui::CalcTextSize(pin.name.c_str()).x);
						ImGui::PopItemWidth();
					}
					ImNodes::EndInputAttribute();
					ImNodes::PopColorStyle();
				}
				else
				{
					ImNodes::PushColorStyle(ImNodesCol_Pin, m_color[pin.type]);
					ImNodes::BeginOutputAttribute(static_cast<int32_t>(pin_handle));
					const float label_width = ImGui::CalcTextSize(pin.name.c_str()).x;
					ImGui::Indent(node_width - label_width);
					ImGui::TextUnformatted(pin.name.c_str());
					ImNodes::EndOutputAttribute();
					ImNodes::PopColorStyle();
				}
			}
			ImNodes::EndNode();
		}
	}

	void DrawEdges(Resource<ResourceType::RenderPipeline> *resource)
	{
		for (auto &[dst, src] : resource->GetDesc().GetEdges())
		{
			const auto &src_node = resource->GetDesc().GetPass(src);
			const auto &dst_node = resource->GetDesc().GetPass(dst);

			if ((src == src_node.GetHandle()) &&
			    (dst == dst_node.GetHandle()))
			{
				ImNodes::PushColorStyle(ImNodesCol_Link, m_color[RenderPassPin::Type::Unknown]);
				ImNodes::Link(static_cast<int32_t>(Hash(src, dst)), -static_cast<int32_t>(src), static_cast<int32_t>(dst));
				ImNodes::PopColorStyle();
			}
			else if ((src != src_node.GetHandle()) &&
			         (dst != dst_node.GetHandle()))
			{
				auto type = src_node.GetPin(src).type;
				ImNodes::PushColorStyle(ImNodesCol_Link, m_color[type]);
				ImNodes::Link(static_cast<int32_t>(Hash(src, dst)), static_cast<int32_t>(src), static_cast<int32_t>(dst));
				ImNodes::PopColorStyle();
			}
		}
	}

	void SetRenderPipeline(Resource<ResourceType::RenderPipeline> *resource, ResourceManager *manager)
	{
		ImGui::PushItemWidth(100.f);
		char buf[128] = {0};
		std::memcpy(buf, m_pipeline_name.data(), sizeof(buf));
		if (ImGui::InputText("##NewRenderPipeline", buf, sizeof(buf)))
		{
			m_pipeline_name = buf;
		}
		ImGui::PopItemWidth();

		ImGui::SameLine();

		if (!m_pipeline_name.empty() && !resource)
		{
			if (ImGui::Button("New Render Pipeline"))
			{
				RenderGraphDesc desc;
				desc.SetName(m_pipeline_name);
				manager->Add<ResourceType::RenderPipeline>(p_editor->GetRHIContext(), m_pipeline_name, std::move(desc));
				resource = manager->Get<ResourceType::RenderPipeline>(m_pipeline_name);
			}
		}
		else
		{
			ImGui::Text("Render Pipeline Name");
		}
	}

	void DragDropResource()
	{
		if (ImGui::BeginDragDropTarget())
		{
			if (const auto *pay_load = ImGui::AcceptDragDropPayload("RenderPipeline"))
			{
				m_pipeline_name = static_cast<const char *>(pay_load->Data);
				auto *resource  = p_editor->GetRenderer()->GetResourceManager()->Get<ResourceType::RenderPipeline>(m_pipeline_name);
				if (resource)
				{
					ImNodes::LoadCurrentEditorStateFromIniString(resource->GetLayout().data(), resource->GetLayout().length());
					auto &desc = resource->GetDesc();
					for (auto &[node_handle, node] : desc.GetPasses())
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

	void DrawInspector()
	{
		ImGui::BeginChild("Render Graph Inspector", ImVec2(0, 0), false, ImGuiWindowFlags_HorizontalScrollbar);

		if (ImGui::GetScrollY() >= ImGui::GetScrollMaxY())
		{
			ImGui::SetScrollHereY(1.0f);
		}

		ImGui::EndChild();
	}

  private:
	ImNodesEditorContext *m_context = nullptr;

	std::string m_pipeline_name = "";

	size_t m_current_handle = 0;

	std::vector<int32_t> m_select_nodes;
	std::vector<int32_t> m_select_links;

	std::vector<int32_t> m_new_nodes;

	std::vector<std::tuple<int32_t, int32_t, uint32_t>> m_edges;

	std::map<RenderPassPin::Type, uint32_t> m_color = {
	    {RenderPassPin::Type::Buffer, IM_COL32(0, 128, 0, 255)},
	    {RenderPassPin::Type::Texture, IM_COL32(128, 0, 0, 255)},
	    {RenderPassPin::Type::Unknown, IM_COL32(255, 255, 255, 255)},
	};
};

extern "C"
{
	EXPORT_API RenderGraphEditor *Create(Editor *editor, ImGuiContext *context)
	{
		ImGui::SetCurrentContext(context);
		return new RenderGraphEditor(editor);
	}
}