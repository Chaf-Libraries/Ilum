#include "RenderGraphEditor.hpp"
#include "Editor/Editor.hpp"

#include <Core/Path.hpp>
#include <ImGui/ImGuiHelper.hpp>
#include <RenderCore/RenderGraph/RenderGraph.hpp>
#include <RenderCore/RenderGraph/RenderGraphBuilder.hpp>
#include <Renderer/Renderer.hpp>
#include <Resource/ResourceManager.hpp>

#include <cereal/cereal.hpp>
#include <cereal/types/array.hpp>
#include <cereal/types/map.hpp>
#include <cereal/types/string.hpp>
#include <cereal/types/unordered_map.hpp>
#include <cereal/types/vector.hpp>

#include <CodeGeneration/Meta/RHIMeta.hpp>
#include <CodeGeneration/Meta/RenderCoreMeta.hpp>
#include <CodeGeneration/Meta/RendererMeta.hpp>
#include <Core/Macro.hpp>

#include <imnodes.h>
#pragma warning(push, 0)
#include <nfd.h>
#pragma warning(pop)

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

				// Erase node
				for (auto node : selected_nodes)
				{
					bool erase = false;

					for (auto iter = m_desc.passes.begin(); iter != m_desc.passes.end();)
					{
						if (static_cast<int32_t>(iter->first.GetHandle()) == node)
						{
							for (auto &[pass_handle, pass] : m_desc.passes)
							{
								if (pass.prev_pass == RGHandle(node))
								{
									pass.prev_pass = RGHandle();
								}
							}
							iter  = m_desc.passes.erase(iter);
							erase = true;
							break;
						}
						else
						{
							iter++;
						}
					}

					for (auto iter = m_desc.textures.begin(); iter != m_desc.textures.end() && !erase;)
					{
						if (static_cast<int32_t>(iter->first.GetHandle()) == node)
						{
							for (auto &[pass_handle, pass] : m_desc.passes)
							{
								for (auto &[name, resource] : pass.resources)
								{
									if (resource.handle == node)
									{
										resource.handle = RGHandle();
									}
								}
							}
							iter  = m_desc.textures.erase(iter);
							erase = true;
							break;
						}
						else
						{
							iter++;
						}
					}

					for (auto iter = m_desc.buffers.begin(); iter != m_desc.buffers.end() && !erase;)
					{
						if (static_cast<int32_t>(iter->first.GetHandle()) == node)
						{
							for (auto &[pass_handle, pass] : m_desc.passes)
							{
								for (auto &[name, resource] : pass.resources)
								{
									if (resource.handle == node)
									{
										resource.handle = RGHandle();
									}
								}
							}
							iter  = m_desc.buffers.erase(iter);
							erase = true;
							break;
						}
						else
						{
							iter++;
						}
					}
				}

				// Erase links
				for (auto link : selected_links)
				{
					for (auto &[handle, pass] : m_desc.passes)
					{
						bool found = false;

						// Pass - Pass
						if (static_cast<int32_t>(Hash(GetPinID(pass.prev_pass, ResourcePinType::PassOut), GetPinID(&pass.prev_pass, PassPinType::PassIn))) == link)
						{
							pass.prev_pass = RGHandle();
							found          = true;
						}

						for (auto &[name, resource] : pass.resources)
						{
							if (static_cast<int32_t>(Hash(GetPinID(resource.handle, ResourcePinType::TextureRead), GetPinID(&resource.handle, PassPinType::PassTexture))) == link ||
							    static_cast<int32_t>(Hash(GetPinID(resource.handle, ResourcePinType::BufferRead), GetPinID(&resource.handle, PassPinType::PassBuffer))) == link ||
							    static_cast<int32_t>(Hash(GetPinID(resource.handle, ResourcePinType::TextureWrite), GetPinID(&resource.handle, PassPinType::PassTexture))) == link ||
							    static_cast<int32_t>(Hash(GetPinID(resource.handle, ResourcePinType::BufferWrite), GetPinID(&resource.handle, PassPinType::PassBuffer))) == link)
							{
								resource.handle = RGHandle();
								found           = true;
							}
						}

						if (found)
						{
							break;
						}
					}
				}
			}
			ImGui::EndPopup();
		}
	}

	std::vector<std::tuple<int32_t, int32_t, uint32_t>> edges;

	// Draw pass nodes
	{
		for (auto &[handle, pass] : m_desc.passes)
		{
			const float node_width = 200.0f;

			ImNodes::PushColorStyle(ImNodesCol_TitleBar, IM_COL32(0, 0, 125, 125));

			ImNodes::BeginNode(static_cast<int32_t>(handle.GetHandle()));
			ImNodes::BeginNodeTitleBar();
			ImGui::Text(pass.name.c_str());
			ImNodes::EndNodeTitleBar();

			// Draw exec pin
			{
				ImNodes::PushColorStyle(ImNodesCol_Pin, IM_COL32(255, 255, 255, 255));
				ImNodes::PushColorStyle(ImNodesCol_PinHovered, IM_COL32(255, 255, 255, 255));
				ImNodes::PushColorStyle(ImNodesCol_Link, IM_COL32(255, 255, 255, 255));

				int32_t exc_pin_id = static_cast<int32_t>((uint64_t) &handle);

				// In attribute
				ImNodes::BeginInputAttribute(GetPinID(&pass.prev_pass, PassPinType::PassIn));
				ImGui::TextUnformatted("In");
				ImNodes::EndInputAttribute();

				if (pass.prev_pass.IsValid())
				{
					edges.emplace_back(std::make_tuple(GetPinID(pass.prev_pass, ResourcePinType::PassOut), GetPinID(&pass.prev_pass, PassPinType::PassIn), IM_COL32(255, 255, 255, 255)));
				}

				ImGui::SameLine();

				// Out attribute
				ImNodes::BeginOutputAttribute(GetPinID(handle, ResourcePinType::PassOut));
				const float label_width = ImGui::CalcTextSize("OutIn ").x;
				ImGui::Indent(node_width - label_width);
				ImGui::TextUnformatted("Out");
				ImNodes::EndOutputAttribute();

				ImNodes::PopColorStyle();
				ImNodes::PopColorStyle();
				ImNodes::PopColorStyle();
			}

			// Draw pin
			for (auto &[name, resource] : pass.resources)
			{
				if (resource.type == RenderResourceDesc::Type::Texture)
				{
					ImNodes::PushColorStyle(ImNodesCol_Pin, IM_COL32(125, 0, 0, 255));
					ImNodes::PushColorStyle(ImNodesCol_PinHovered, IM_COL32(125, 0, 0, 255));
					if (resource.attribute == RenderResourceDesc::Attribute::Read)
					{
						ImNodes::BeginInputAttribute(GetPinID(&resource.handle, PassPinType::PassTexture));
						ImGui::TextUnformatted(name.c_str());
						ImNodes::EndInputAttribute();
						if (resource.handle.IsValid())
						{
							edges.emplace_back(std::make_tuple(GetPinID(resource.handle, ResourcePinType::TextureRead), GetPinID(&resource.handle, PassPinType::PassTexture), IM_COL32(125, 0, 0, 255)));
						}
					}
					else
					{
						ImNodes::BeginOutputAttribute(GetPinID(&resource.handle, PassPinType::PassTexture));
						const float label_width = ImGui::CalcTextSize(name.c_str()).x;
						ImGui::Indent(node_width - label_width);
						ImGui::TextUnformatted(name.c_str());
						ImNodes::EndOutputAttribute();
						if (resource.handle.IsValid())
						{
							edges.emplace_back(std::make_tuple(GetPinID(resource.handle, ResourcePinType::TextureWrite), GetPinID(&resource.handle, PassPinType::PassTexture), IM_COL32(125, 0, 0, 255)));
						}
					}
					ImNodes::PopColorStyle();
					ImNodes::PopColorStyle();
				}
				else
				{
					ImNodes::PushColorStyle(ImNodesCol_Pin, IM_COL32(0, 125, 0, 255));
					ImNodes::PushColorStyle(ImNodesCol_PinHovered, IM_COL32(0, 125, 0, 255));
					if (resource.attribute == RenderResourceDesc::Attribute::Read)
					{
						ImNodes::BeginInputAttribute(GetPinID(&resource.handle, PassPinType::PassBuffer));
						ImGui::TextUnformatted(name.c_str());
						ImNodes::EndInputAttribute();
						if (resource.handle.IsValid())
						{
							edges.emplace_back(std::make_tuple(GetPinID(resource.handle, ResourcePinType::BufferRead), GetPinID(&resource.handle, PassPinType::PassBuffer), IM_COL32(125, 0, 0, 255)));
						}
					}
					else
					{
						ImNodes::BeginOutputAttribute(GetPinID(&resource.handle, PassPinType::PassTexture));
						const float label_width = ImGui::CalcTextSize(name.c_str()).x;
						ImGui::Indent(node_width - label_width);
						ImGui::TextUnformatted(name.c_str());
						ImNodes::EndOutputAttribute();
						if (resource.handle.IsValid())
						{
							edges.emplace_back(std::make_tuple(GetPinID(resource.handle, ResourcePinType::BufferWrite), GetPinID(&resource.handle, PassPinType::PassBuffer), IM_COL32(125, 0, 0, 255)));
						}
					}
					ImNodes::PopColorStyle();
					ImNodes::PopColorStyle();

					if (resource.handle.IsValid())
					{
						edges.emplace_back(std::make_tuple(GetPinID(resource.handle, ResourcePinType::BufferRead), GetPinID(&resource.handle, PassPinType::PassBuffer), IM_COL32(125, 0, 0, 255)));
					}
				}
			}

			ImNodes::EndNode();

			ImNodes::PopColorStyle();
		}
	}

	// Draw Texture Node
	{
		for (auto &[handle, texture] : m_desc.textures)
		{
			const float node_width = 100.0f;

			ImNodes::PushColorStyle(ImNodesCol_TitleBar, IM_COL32(125, 0, 0, 125));
			ImNodes::PushColorStyle(ImNodesCol_Pin, IM_COL32(125, 0, 0, 255));
			ImNodes::PushColorStyle(ImNodesCol_PinHovered, IM_COL32(125, 0, 0, 255));
			ImNodes::BeginNode(static_cast<int32_t>(handle.GetHandle()));
			ImNodes::BeginNodeTitleBar();
			ImGui::Text(texture.name.c_str());
			ImNodes::EndNodeTitleBar();

			ImNodes::BeginInputAttribute(GetPinID(handle, ResourcePinType::TextureWrite));
			ImGui::TextUnformatted("Write");
			ImNodes::EndInputAttribute();

			ImGui::SameLine();

			ImNodes::BeginOutputAttribute(GetPinID(handle, ResourcePinType::TextureRead));
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
			ImNodes::PushColorStyle(ImNodesCol_Pin, IM_COL32(0, 125, 0, 255));
			ImNodes::PushColorStyle(ImNodesCol_PinHovered, IM_COL32(0, 125, 0, 255));
			ImNodes::BeginNode(static_cast<int32_t>(handle.GetHandle()));
			ImNodes::BeginNodeTitleBar();
			ImGui::Text(buffer.name.c_str());
			ImNodes::EndNodeTitleBar();

			ImNodes::BeginInputAttribute(GetPinID(handle, ResourcePinType::BufferWrite));
			ImGui::TextUnformatted("Write");
			ImNodes::EndInputAttribute();

			ImGui::SameLine();

			ImNodes::BeginOutputAttribute(GetPinID(handle, ResourcePinType::BufferRead));
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
		for (auto &[src, dst, color] : edges)
		{
			ImNodes::PushColorStyle(ImNodesCol_Link, color);
			ImNodes::Link(static_cast<int32_t>(Hash(src, dst)), src, dst);
			ImNodes::PopColorStyle();
		}
	}

	ImNodes::MiniMap(0.1f);
	ImNodes::EndNodeEditor();

	if (ImGui::BeginDragDropTarget())
	{
		if (const auto *pay_load = ImGui::AcceptDragDropPayload("RenderGraph"))
		{
			ASSERT(pay_load->DataSize == sizeof(std::string));
			std::string uuid = *static_cast<std::string *>(pay_load->Data);
			auto       *meta = p_editor->GetRenderer()->GetResourceManager()->GetRenderGraph(uuid);
			if (meta)
			{
				std::ifstream is("Asset/Meta/" + uuid + ".meta", std::ios::binary);
				InputArchive  archive(is);
				std::string   filename;
				archive(ResourceType::RenderGraph, uuid, filename);
				std::string editor_state = "";
				archive(m_desc, editor_state);
				ImNodes::LoadCurrentEditorStateFromIniString(editor_state.data(), editor_state.size());
				m_need_compile = true;
			}
		}
	}

	// Create New Edges
	{
		int32_t src = 0, dst = 0;
		if (ImNodes::IsLinkCreated(&src, &dst))
		{
			ResourcePinType resource_type = ResourcePinType::None;
			PassPinType     pass_type     = PassPinType::None;

			RGHandle  resource_handle;
			RGHandle *pass_handle = nullptr;

			if (m_resource_pin.find(src) != m_resource_pin.end() &&
			    m_pass_pin.find(dst) != m_pass_pin.end())
			{
				resource_handle = m_resource_pin[src].first;
				resource_type   = m_resource_pin[src].second;

				pass_handle = m_pass_pin[dst].first;
				pass_type   = m_pass_pin[dst].second;
			}
			else if (m_resource_pin.find(dst) != m_resource_pin.end() &&
			         m_pass_pin.find(src) != m_pass_pin.end())
			{
				resource_handle = m_resource_pin[dst].first;
				resource_type   = m_resource_pin[dst].second;

				pass_handle = m_pass_pin[src].first;
				pass_type   = m_pass_pin[src].second;
			}

			if (ValidLink(resource_type, pass_type))
			{
				if (pass_handle)
				{
					*pass_handle = resource_handle;
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
				m_need_compile |= ImGui::EditVariant(texture);
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
				m_need_compile |= ImGui::EditVariant(buffer);
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
					m_need_compile |= ImGui::EditVariant(pass.config);
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

	if (ImGui::BeginMenuBar())
	{
		if (ImGui::BeginMenu("File"))
		{
			if (ImGui::MenuItem("Load"))
			{
				char *path = nullptr;
				if (NFD_OpenDialog("rg", Path::GetInstance().GetCurrent(false).c_str(), &path) == NFD_OKAY)
				{
					std::string editor_state = "";
					DESERIALIZE(path, m_desc, editor_state);
					ImNodes::LoadCurrentEditorStateFromIniString(editor_state.data(), editor_state.size());

					// Update max current id
					m_current_handle = 0;

					for (auto &[handle, pass] : m_desc.passes)
					{
						m_current_handle = std::max(handle.GetHandle(), m_current_handle);
					}

					for (auto &[handle, texture] : m_desc.textures)
					{
						m_current_handle = std::max(handle.GetHandle(), m_current_handle);
					}

					for (auto &[handle, buffer] : m_desc.buffers)
					{
						m_current_handle = std::max(handle.GetHandle(), m_current_handle);
					}

					m_current_handle++;
					m_need_compile = true;

					free(path);
				}
			}

			if (ImGui::MenuItem("Save"))
			{
				char *path = nullptr;
				if (NFD_SaveDialog("rg", Path::GetInstance().GetCurrent(false).c_str(), &path) == NFD_OKAY)
				{
					std::string dir      = Path::GetInstance().GetFileDirectory(path);
					std::string filename = Path::GetInstance().GetFileName(path, false);
					SERIALIZE(dir + filename + ".rg", m_desc, std::string(ImNodes::SaveCurrentEditorStateToIniString()));
					// Save as engine meta
					{
						std::string   uuid = std::to_string(Hash(filename));
						std::ofstream os("Asset/Meta/" + uuid + ".meta", std::ios::binary);
						OutputArchive archive(os);
						archive(ResourceType::RenderGraph, uuid, filename, m_desc, std::string(ImNodes::SaveCurrentEditorStateToIniString()));
						RenderGraphMeta meta;
						meta.name = filename;
						meta.uuid = uuid;
						p_editor->GetRenderer()->GetResourceManager()->AddRenderGraphMeta(std::move(meta));
					}
					free(path);
				}
			}

			ImGui::EndMenu();
		}

		if (ImGui::BeginMenu("Add"))
		{
			if (ImGui::BeginMenu("Pass"))
			{
				for (auto &type : rttr::type::get_types())
				{
					if (type.get_metadata("RenderPass"))
					{
						std::string pass_name = type.get_metadata("RenderPass").get_value<std::string>();
						if (type.get_metadata("Category"))
						{
							if (ImGui::BeginMenu(type.get_metadata("Category").get_value<std::string>().c_str()))
							{
								if (ImGui::MenuItem(pass_name.c_str()))
								{
									auto pass = type.create();
									m_desc.passes.emplace(
									    RGHandle(m_current_handle++),
									    rttr::type::get(pass).get_method("CreateDesc").invoke(pass).convert<RenderPassDesc>());
									m_need_compile = true;
								}
								ImGui::EndMenu();
							}
						}
						else
						{
							if (ImGui::MenuItem(pass_name.c_str()))
							{
								auto pass = type.create();
								m_desc.passes.emplace(
								    RGHandle(m_current_handle++),
								    rttr::type::get(pass).get_method("CreateDesc").invoke(pass).convert<RenderPassDesc>());
								m_need_compile = true;
							}
						}
					}
				}
				ImGui::EndMenu();
			}

			if (ImGui::MenuItem("Texture"))
			{
				m_desc.textures.emplace(RGHandle(m_current_handle++), TextureDesc{"Texture", 1, 1, 1, 1, 1, 1, RHIFormat::R8G8B8A8_UNORM, RHITextureUsage::Undefined});
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

		if (ImGui::MenuItem("Clear"))
		{
			m_desc.buffers.clear();
			m_desc.textures.clear();
			m_desc.passes.clear();
			m_need_compile = true;
		}

		if (m_need_compile && ImGui::MenuItem("Compile"))
		{
			RenderGraphBuilder builder(p_editor->GetRHIContext());

			auto *renderer     = p_editor->GetRenderer();
			auto  render_graph = builder.Compile(m_desc, renderer);

			if (render_graph)
			{
				renderer->SetRenderGraph(std::move(render_graph));
				m_need_compile = false;
			}
			else
			{
				LOG_INFO("Render Graph Compile Failed!");
			}
		}

		if (!m_need_compile && !m_desc.passes.empty() && ImGui::MenuItem("Reload"))
		{
			p_editor->GetRHIContext()->WaitIdle();
			p_editor->GetRHIContext()->Reset();

			RenderGraphBuilder builder(p_editor->GetRHIContext());

			auto *renderer = p_editor->GetRenderer();
			renderer->Reset();
			auto render_graph = builder.Compile(m_desc, renderer);

			if (render_graph)
			{
				renderer->SetRenderGraph(std::move(render_graph));
				m_need_compile = false;
			}
			else
			{
				LOG_INFO("Render Graph Compile Failed!");
			}
		}

		ImGui::EndMenuBar();
	}

	ImGui::End();
}

int32_t RenderGraphEditor::GetPinID(RGHandle *handle, PassPinType pin)
{
	int32_t id     = static_cast<int32_t>(Hash(handle, pin));
	m_pass_pin[id] = std::make_pair(handle, pin);
	return id;
}

int32_t RenderGraphEditor::GetPinID(const RGHandle &handle, ResourcePinType pin)
{
	int32_t id         = static_cast<int32_t>(Hash(handle.GetHandle(), pin));
	m_resource_pin[id] = std::make_pair(handle, pin);
	return id;
}

bool RenderGraphEditor::ValidLink(ResourcePinType resource, PassPinType pass)
{
	switch (resource)
	{
		case ResourcePinType::TextureWrite:
		case ResourcePinType::TextureRead:
			return pass == PassPinType::PassTexture;
		case ResourcePinType::BufferWrite:
		case ResourcePinType::BufferRead:
			return pass == PassPinType::PassBuffer;
		case ResourcePinType::PassOut:
			return pass == PassPinType::PassIn;
		default:
			break;
	}

	return false;
}
}        // namespace Ilum