#include "RenderGraphEditor.hpp"

#include "Editor/Editor.hpp"

#include <RenderCore/RenderGraph/RenderGraph.hpp>

#include <Renderer/RenderPass.hpp>

#include <ImGui/ImGuiHelper.hpp>

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
	static auto editor_context = ImNodes::EditorContextCreate();

	ImGui::Begin(m_name.c_str(), &m_active, ImGuiWindowFlags_MenuBar);

	ImNodes::EditorContextSet(editor_context);

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
						m_desc.passes.emplace(
						    RGHandle(m_current_handle++),
						    rttr::type::invoke(fmt::format("{}_Desc", pass_name->name).c_str(), {m_current_handle}).get_value<RenderPassDesc>());
						m_need_compile = true;
					}
					pass_name = pass_name->next;
				}
				ImGui::EndMenu();
			}

			if (ImGui::MenuItem("Texture"))
			{
				m_desc.textures.emplace(
				    RGHandle(m_current_handle++),
				    TextureDesc{"Texture"});
				m_need_compile = true;
			}

			if (ImGui::MenuItem("Buffer"))
			{
				m_desc.buffers.emplace(
				    RGHandle(m_current_handle++),
				    BufferDesc{"Buffer"});
				m_need_compile = true;
			}
			ImGui::EndMenu();
		}

		if (m_need_compile && ImGui::MenuItem("Compile"))
		{
			RenderGraphBuilder builder(p_editor->GetRHIContext());

			auto *renderer = p_editor->GetRenderer();

			if (builder.Compile(m_desc, renderer))
			{
				m_need_compile = false;
			}
			else
			{
				LOG_INFO("Render Graph Compile Failed!");
			}
		}
		ImGui::EndMenuBar();
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
				m_need_compile = true;

				std::set<RGHandle> deprecated_handles;

				// Remove Passes
				for (auto &node : selected_nodes)
				{
					RGHandle deprecated_handle(static_cast<size_t>(node));
					deprecated_handles.insert(deprecated_handle);

					if (m_desc.passes.find(deprecated_handle) != m_desc.passes.end())
					{
						for (auto& write : m_desc.passes[deprecated_handle].writes)
						{
							deprecated_handles.insert(write.second.handle);
						}
						for (auto &read : m_desc.passes[deprecated_handle].reads)
						{
							deprecated_handles.insert(read.second.handle);
						}
						m_desc.passes.erase(deprecated_handle);
					}
					else if (m_desc.textures.find(deprecated_handle) != m_desc.textures.end())
					{
						m_desc.textures.erase(deprecated_handle);
					}
					else if (m_desc.buffers.find(deprecated_handle) != m_desc.buffers.end())
					{
						m_desc.buffers.erase(deprecated_handle);
					}
				}

				// Remove Edges
				for (auto iter = m_desc.edges.begin(); iter != m_desc.edges.end();)
				{
					size_t hash = 0;
					HashCombine(hash, iter->first, iter->second);

					auto [src_handle, dst_handle] = RenderGraphDesc::DecodeEdge(iter->first, iter->second);

					if (deprecated_handles.find(src_handle) != deprecated_handles.end() ||
					    deprecated_handles.find(dst_handle) != deprecated_handles.end())
					{
						iter = m_desc.edges.erase(iter);
						continue;
					}

					bool erase = false;

					for (auto &link : selected_links)
					{
						if (link == static_cast<int32_t>(hash))
						{
							iter  = m_desc.edges.erase(iter);
							erase = true;
							break;
						}
					}

					if (!erase)
					{
						iter++;
					}
				}
			}
			ImGui::EndPopup();
		}
	}

	// Draw pass nodes
	{
		for (auto &[handle, pass] : m_desc.passes)
		{
			const float node_width = 200.0f;

			ImNodes::PushColorStyle(ImNodesCol_TitleBar, IM_COL32(0, 0, 125, 125));
			ImNodes::PushColorStyle(ImNodesCol_TitleBarHovered, IM_COL32(0, 0, 125, 125));
			ImNodes::PushColorStyle(ImNodesCol_TitleBarSelected, IM_COL32(0, 0, 125, 125));

			ImNodes::BeginNode(static_cast<int32_t>(handle.GetHandle()));
			ImNodes::BeginNodeTitleBar();
			ImGui::Text(pass.name.c_str());
			ImNodes::EndNodeTitleBar();

			// Draw read pin
			for (auto &[name, read] : pass.reads)
			{
				if (read.type == RenderPassDesc::ResourceInfo::Type::Texture)
				{
					ImNodes::BeginInputAttribute(static_cast<int32_t>(read.handle.GetHandle()) * 2, ImNodesPinShape_CircleFilled);
					ImGui::TextUnformatted(name.c_str());
					ImNodes::EndInputAttribute();
				}
				else
				{
					ImNodes::BeginInputAttribute(static_cast<int32_t>(read.handle.GetHandle()) * 2, ImNodesPinShape_TriangleFilled);
					ImGui::TextUnformatted(name.c_str());
					ImNodes::EndInputAttribute();
				}
			}

			// Draw write pin
			for (auto &[name, write] : pass.writes)
			{
				if (write.type == RenderPassDesc::ResourceInfo::Type::Texture)
				{
					ImNodes::BeginOutputAttribute(static_cast<int32_t>(write.handle.GetHandle()) * 2, ImNodesPinShape_CircleFilled);
					const float label_width = ImGui::CalcTextSize(name.c_str()).x;
					ImGui::Indent(node_width - label_width);
					ImGui::TextUnformatted(name.c_str());
					ImNodes::EndOutputAttribute();
				}
				else
				{
					ImNodes::BeginOutputAttribute(static_cast<int32_t>(write.handle.GetHandle()) * 2, ImNodesPinShape_TriangleFilled);
					const float label_width = ImGui::CalcTextSize(name.c_str()).x;
					ImGui::Indent(node_width - label_width);
					ImGui::TextUnformatted(name.c_str());
					ImNodes::EndOutputAttribute();
				}
			}

			ImNodes::EndNode();

			ImNodes::PopColorStyle();
			ImNodes::PopColorStyle();
			ImNodes::PopColorStyle();
		}
	}

	// Draw Texture Node
	{
		for (auto &[handle, texture] : m_desc.textures)
		{
			const float node_width = 100.0f;

			ImNodes::PushColorStyle(ImNodesCol_TitleBar, IM_COL32(125, 0, 0, 125));
			ImNodes::PushColorStyle(ImNodesCol_TitleBarHovered, IM_COL32(125, 0, 0, 125));
			ImNodes::PushColorStyle(ImNodesCol_TitleBarSelected, IM_COL32(125, 0, 0, 125));
			ImNodes::BeginNode(static_cast<int32_t>(handle.GetHandle()));
			ImNodes::BeginNodeTitleBar();
			ImGui::Text(texture.name.c_str());
			ImNodes::EndNodeTitleBar();

			ImNodes::BeginInputAttribute(static_cast<int32_t>(handle.GetHandle()) * 2, ImNodesPinShape_CircleFilled);
			ImGui::TextUnformatted("Write");
			ImNodes::EndInputAttribute();

			ImGui::SameLine();

			ImNodes::BeginOutputAttribute(static_cast<int32_t>(handle.GetHandle()) * 2 + 1, ImNodesPinShape_CircleFilled);
			const float label_width = ImGui::CalcTextSize("Read").x;
			ImGui::Indent(node_width - label_width);
			ImGui::TextUnformatted("Read");
			ImNodes::EndOutputAttribute();

			ImNodes::EndNode();

			ImNodes::PopColorStyle();
			ImNodes::PopColorStyle();
			ImNodes::PopColorStyle();
		}
	}

	// Draw Buffer Node
	{
		for (auto &[handle, buffer] : m_desc.buffers)
		{
			const float node_width = 100.0f;

			ImNodes::PushColorStyle(ImNodesCol_TitleBar, IM_COL32(0, 125, 0, 125));
			ImNodes::PushColorStyle(ImNodesCol_TitleBarHovered, IM_COL32(0, 125, 0, 125));
			ImNodes::PushColorStyle(ImNodesCol_TitleBarSelected, IM_COL32(0, 125, 0, 125));
			ImNodes::BeginNode(static_cast<int32_t>(handle.GetHandle()));
			ImNodes::BeginNodeTitleBar();
			ImGui::Text(buffer.name.c_str());
			ImNodes::EndNodeTitleBar();

			ImNodes::BeginInputAttribute(static_cast<int32_t>(handle.GetHandle()) * 2, ImNodesPinShape_TriangleFilled);
			ImGui::TextUnformatted("Write");
			ImNodes::EndInputAttribute();

			ImGui::SameLine();

			ImNodes::BeginOutputAttribute(static_cast<int32_t>(handle.GetHandle()) * 2 + 1, ImNodesPinShape_TriangleFilled);
			const float label_width = ImGui::CalcTextSize("Read").x;
			ImGui::Indent(node_width - label_width);
			ImGui::TextUnformatted("Read");
			ImNodes::EndOutputAttribute();

			ImNodes::EndNode();
			ImNodes::PopColorStyle();
			ImNodes::PopColorStyle();
			ImNodes::PopColorStyle();
		}
	}

	// Draw Edges
	{
		for (auto &[src, dst] : m_desc.edges)
		{
			size_t hash = 0;
			HashCombine(hash, src, dst);
			ImNodes::Link(static_cast<int32_t>(hash), static_cast<int32_t>(src), static_cast<int32_t>(dst));
		}
	}

	ImNodes::MiniMap(0.1f);
	ImNodes::EndNodeEditor();

	// Create New Edges
	{
		int32_t src = 0, dst = 0;
		if (ImNodes::IsLinkCreated(&src, &dst))
		{
			bool src_texture  = false;
			bool src_buffer   = false;
			bool src_resource = false;

			bool dst_texture  = false;
			bool dst_buffer   = false;
			bool dst_resource = false;

			auto [src_handle, dst_handle] = RenderGraphDesc::DecodeEdge(static_cast<size_t>(src), static_cast<size_t>(dst));

			src_texture  = m_desc.textures.find(src_handle) != m_desc.textures.end();
			src_buffer   = m_desc.buffers.find(src_handle) != m_desc.buffers.end();
			src_resource = src_texture || src_buffer;

			dst_texture  = m_desc.textures.find(dst_handle) != m_desc.textures.end();
			dst_buffer   = m_desc.buffers.find(dst_handle) != m_desc.buffers.end();
			dst_resource = dst_texture || dst_buffer;

			// Resource can only be written by one pass, if want more, use copy pass
			// A pass output can only be a single texture
			if (dst_resource)
			{
				for (auto iter = m_desc.edges.begin(); iter != m_desc.edges.end();)
				{
					if (dst == iter->second || src == iter->first)
					{
						iter = m_desc.edges.erase(iter);
						m_need_compile = true;
					}
					else
					{
						iter++;
					}
				}
			}

			if (src_resource != dst_resource)
			{
				bool finish = false;

				if (src_resource)
				{
					for (auto &[pass_handle, pass] : m_desc.passes)
					{
						for (auto &[name, read] : pass.reads)
						{
							if (src_buffer && read.type == RenderPassDesc::ResourceInfo::Type::Buffer)
							{
								m_desc.edges.insert(std::make_pair(src, dst));
								m_need_compile = true;
								finish         = true;
								break;
							}
							else if (src_texture && read.type == RenderPassDesc::ResourceInfo::Type::Texture)
							{
								m_desc.edges.insert(std::make_pair(src, dst));
								m_need_compile = true;
								finish         = true;
								break;
							}
						}
						if (finish)
						{
							break;
						}
					}
				}

				if (dst_resource)
				{
					for (auto &[pass_handle, pass] : m_desc.passes)
					{
						for (auto &[name, write] : pass.writes)
						{
							if (dst_buffer && write.type == RenderPassDesc::ResourceInfo::Type::Buffer)
							{
								m_desc.edges.insert(std::make_pair(src, dst));
								m_need_compile = true;
								finish         = true;
								break;
							}
							else if (dst_texture && write.type == RenderPassDesc::ResourceInfo::Type::Texture)
							{
								m_desc.edges.insert(std::make_pair(src, dst));
								m_need_compile = true;
								finish         = true;
								break;
							}
						}
						if (finish)
						{
							break;
						}
					}
				}
			}
		}
	}

	ImGui::NextColumn();

	// Property Setting
	{
		ImGui::BeginChild("Render Graph Inspector", ImVec2(0, 0), false, ImGuiWindowFlags_HorizontalScrollbar);

		ImGui::PushItemWidth(ImGui::GetColumnWidth(1) * 0.7f);

		// Texture Inspector
		for (auto &[handle, texture] : m_desc.textures)
		{
			if (ImNodes::IsNodeSelected(static_cast<int32_t>(handle.GetHandle())))
			{
				ImGui::PushID(static_cast<int32_t>(handle.GetHandle()));
				ImGui::Text("Texture - %s", texture.name.c_str());

				{
					char buf[64] = {0};
					std::memcpy(buf, texture.name.data(), sizeof(buf));
					if (ImGui::InputText("##Tag", buf, sizeof(buf)), ImGuiInputTextFlags_AutoSelectAll)
					{
						texture.name   = std::string(buf);
						m_need_compile = true;
					}
				}

				m_need_compile |= ImGui::DragInt("Width", reinterpret_cast<int32_t *>(&texture.width), 0.1f, 0);
				m_need_compile |= ImGui::DragInt("Height", reinterpret_cast<int32_t *>(&texture.height), 0.1f, 0);
				m_need_compile |= ImGui::DragInt("Depth", reinterpret_cast<int32_t *>(&texture.depth), 0.1f, 0);
				m_need_compile |= ImGui::DragInt("Mips", reinterpret_cast<int32_t *>(&texture.mips), 0.1f, 0);
				m_need_compile |= ImGui::DragInt("Layers", reinterpret_cast<int32_t *>(&texture.layers), 0.1f, 0);

				const char *const formats[] = {
				    "Undefined",
				    "R8G8B8A8_UNORM",
				    "B8G8R8A8_UNORM",
				    "R16_UINT",
				    "R16_SINT",
				    "R16_FLOAT",
				    "R16G16_UINT",
				    "R16G16_SINT",
				    "R16G16_FLOAT",
				    "R16G16B16A16_UINT",
				    "R16G16B16A16_SINT",
				    "R16G16B16A16_FLOAT",
				    "R32_UINT",
				    "R32_SINT",
				    "R32_FLOAT",
				    "R32G32_UINT",
				    "R32G32_SINT",
				    "R32G32_FLOAT",
				    "R32G32B32_UINT",
				    "R32G32B32_SINT",
				    "R32G32B32_FLOAT",
				    "R32G32B32A32_UINT",
				    "R32G32B32A32_SINT",
				    "R32G32B32A32_FLOAT",
				    "D32_FLOAT",
				    "D24_UNORM_S8_UINT"};

				m_need_compile |= ImGui::Combo("Format", reinterpret_cast<int32_t *>(&texture.format), formats, 26);

				ImGui::PopID();
				ImGui::Separator();
			}
		}

		// Buffer Inspector
		for (auto &[handle, buffer] : m_desc.buffers)
		{
			if (ImNodes::IsNodeSelected(static_cast<int32_t>(handle.GetHandle())))
			{
				ImGui::PushID(static_cast<int32_t>(handle.GetHandle()));
				ImGui::Text("Buffer - %s", buffer.name.c_str());

				{
					char buf[64] = {0};
					std::memcpy(buf, buffer.name.data(), sizeof(buf));
					if (ImGui::InputText("##Tag", buf, sizeof(buf)), ImGuiInputTextFlags_AutoSelectAll)
					{
						buffer.name    = std::string(buf);
						m_need_compile = true;
					}
				}

				m_need_compile |= ImGui::DragScalar("Size", ImGuiDataType_U32, &buffer.size, 0.1f, 0);

				ImGui::PopID();
				ImGui::Separator();
			}
		}

		// Pass Inspector
		{
			for (auto &[handle, pass] : m_desc.passes)
			{
				if (ImNodes::IsNodeSelected(static_cast<int32_t>(handle.GetHandle())))
				{
					ImGui::PushID(static_cast<int32_t>(handle.GetHandle()));
					ImGui::Text("Pass - %s", pass.name.c_str());
					m_need_compile |= ImGui::EditVariant(pass.variant);
					ImGui::PopID();
					ImGui::Separator();
				}
			}
		}

		ImGui::PopItemWidth();

		if (ImGui::GetScrollY() >= ImGui::GetScrollMaxY())
		{
			ImGui::SetScrollHereY(1.0f);
		}

		ImGui::EndChild();
	}

	ImGui::End();
}
}        // namespace Ilum