#include <Editor/Editor.hpp>
#include <Editor/Widget.hpp>
#include <RenderGraph/RenderGraph.hpp>
#include <RenderGraph/RenderGraphBuilder.hpp>
#include <Renderer/Renderer.hpp>

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
		if (!ImGui::Begin(m_name.c_str(), nullptr, ImGuiWindowFlags_MenuBar))
		{
			ImGui::End();
			return;
		}

		ImNodes::EditorContextSet(m_context);

		DrawMenu();

		ImGui::Columns(2);
		ImGui::SetColumnWidth(0, ImGui::GetWindowWidth() * 0.2f);

		DrawInspector();

		ImGui::NextColumn();

		HandleSelection();

		ImNodes::BeginNodeEditor();

		PopupWindow();
		DrawTextureNodes();
		DrawBufferNodes();
		DrawPassNode();
		DrawEdges();

		ImNodes::MiniMap(0.1f);
		ImNodes::EndNodeEditor();

		AddEdge();

		ImGui::End();
	}

  private:
	enum class ResourcePinType
	{
		None,
		PassOut,
		TextureWrite,
		TextureRead,
		BufferWrite,
		BufferRead
	};

	enum class PassPinType
	{
		None,
		PassIn,
		PassTexture,
		PassBuffer
	};

  private:
	void DrawMenu()
	{
		if (ImGui::BeginMenuBar())
		{
			if (ImGui::MenuItem("Load"))
			{
				char *path = nullptr;
				if (NFD_OpenDialog("rg", Path::GetInstance().GetCurrent(false).c_str(), &path) == NFD_OKAY)
				{
					m_desc.buffers.clear();
					m_desc.textures.clear();
					m_desc.passes.clear();

					std::string editor_state = "";
					DESERIALIZE(path, m_desc, editor_state);
					ImNodes::LoadCurrentEditorStateFromIniString(editor_state.data(), editor_state.length());

					m_current_handle = 0;

					for (auto &[handle, buffer] : m_desc.buffers)
					{
						m_current_handle = glm::max(m_current_handle, handle.GetHandle());
					}
					for (auto &[handle, texture] : m_desc.textures)
					{
						m_current_handle = glm::max(m_current_handle, handle.GetHandle());
					}
					for (auto &[handle, pass] : m_desc.passes)
					{
						m_current_handle = glm::max(m_current_handle, handle.GetHandle());
					}
					m_current_handle++;
				}
			}

			if (ImGui::MenuItem("Save"))
			{
				char *path = nullptr;
				if (NFD_SaveDialog("rg", Path::GetInstance().GetCurrent(false).c_str(), &path) == NFD_OKAY)
				{
					std::string editor_state = ImNodes::SaveCurrentEditorStateToIniString();
					SERIALIZE(Path::GetInstance().GetFileExtension(path) == ".rg" ? path : std::string(path) + ".rg", m_desc, editor_state);
				}
			}

			if (ImGui::MenuItem("Clear"))
			{
				m_desc.buffers.clear();
				m_desc.textures.clear();
				m_desc.passes.clear();
			}

			if (ImGui::MenuItem("Compile"))
			{
				RenderGraphBuilder builder(p_editor->GetRHIContext());

				auto *renderer     = p_editor->GetRenderer();
				auto  render_graph = builder.Compile(m_desc, renderer);

				if (render_graph)
				{
					renderer->SetRenderGraph(std::move(render_graph));
				}
				else
				{
					LOG_INFO("Render Graph Compile Failed!");
				}
			}

			ImGui::EndMenuBar();
		}
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

	void PopupWindow()
	{
		if (ImGui::BeginPopupContextWindow(0, ImGuiPopupFlags_MouseButtonRight))
		{
			if (ImGui::MenuItem("New Texture"))
			{
				m_desc.textures.emplace(
				    RGHandle(m_current_handle++),
				    TextureDesc{"Texture", 1, 1, 1, 1, 1, 1, RHIFormat::R8G8B8A8_UNORM, RHITextureUsage::Undefined});
				ImNodes::SetNodeScreenSpacePos(static_cast<int32_t>(m_current_handle) - 1, ImGui::GetMousePos());
			}

			if (ImGui::MenuItem("New Buffer"))
			{
				m_desc.buffers.emplace(
				    RGHandle(m_current_handle++),
				    BufferDesc{"Buffer"});
				ImNodes::SetNodeScreenSpacePos(static_cast<int32_t>(m_current_handle) - 1, ImGui::GetMousePos());
			}

			if (ImGui::BeginMenu("New Pass"))
			{
				for (const auto &file : std::filesystem::directory_iterator("./lib/"))
				{
					std::string filename = file.path().filename().string();
					if (std::regex_match(filename, std::regex("(Pass.)(.*)(.dll)")))
					{
						size_t begin = filename.find_first_of('.') + 1;
						size_t end   = filename.find_last_of('.');

						std::string pass_name = filename.substr(begin, end - begin);

						if (ImGui::MenuItem(pass_name.c_str()))
						{
							RenderPassDesc pass_desc;
							PluginManager::GetInstance().Call(filename, "CreateDesc", &pass_desc);
							pass_desc.name = pass_name;
							m_desc.passes.emplace(RGHandle(m_current_handle++), std::move(pass_desc));
							ImNodes::SetNodeScreenSpacePos(static_cast<int32_t>(m_current_handle) - 1, ImGui::GetMousePos());
						}
					}
				}
				ImGui::EndMenu();
			}

			if (!m_select_links.empty() || !m_select_nodes.empty())
			{
				if (ImGui::MenuItem("Remove"))
				{
					// Erase node
					for (auto node : m_select_nodes)
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
					for (auto link : m_select_links)
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
			}

			ImGui::EndPopup();
		}
	}

	void DrawTextureNodes()
	{
		const float node_width = 100.0f;

		ImNodes::PushColorStyle(ImNodesCol_TitleBar, IM_COL32(125, 0, 0, 125));
		ImNodes::PushColorStyle(ImNodesCol_Pin, IM_COL32(125, 0, 0, 255));
		ImNodes::PushColorStyle(ImNodesCol_PinHovered, IM_COL32(125, 0, 0, 255));

		for (auto &[handle, texture] : m_desc.textures)
		{
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
		}

		ImNodes::PopColorStyle();
		ImNodes::PopColorStyle();
		ImNodes::PopColorStyle();
	}

	void DrawBufferNodes()
	{
		const float node_width = 100.0f;

		ImNodes::PushColorStyle(ImNodesCol_TitleBar, IM_COL32(0, 125, 0, 125));
		ImNodes::PushColorStyle(ImNodesCol_Pin, IM_COL32(0, 125, 0, 255));
		ImNodes::PushColorStyle(ImNodesCol_PinHovered, IM_COL32(0, 125, 0, 255));

		for (auto &[handle, buffer] : m_desc.buffers)
		{
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
		}

		ImNodes::PopColorStyle();
		ImNodes::PopColorStyle();
		ImNodes::PopColorStyle();
	}

	void DrawPassNode()
	{
		m_edges.clear();

		ImNodes::PushColorStyle(ImNodesCol_TitleBar, IM_COL32(0, 0, 125, 125));

		for (auto &[handle, pass] : m_desc.passes)
		{
			float node_width = 0.f;

			node_width = std::max(node_width, ImGui::CalcTextSize(pass.name.c_str()).x);
			for (auto &[name, resource] : pass.resources)
			{
				node_width = std::max(node_width, ImGui::CalcTextSize(name.c_str()).x);
			}

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
					m_edges.emplace_back(std::make_tuple(GetPinID(pass.prev_pass, ResourcePinType::PassOut), GetPinID(&pass.prev_pass, PassPinType::PassIn), IM_COL32(255, 255, 255, 255)));
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
							m_edges.emplace_back(std::make_tuple(GetPinID(resource.handle, ResourcePinType::TextureRead), GetPinID(&resource.handle, PassPinType::PassTexture), IM_COL32(125, 0, 0, 255)));
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
							m_edges.emplace_back(std::make_tuple(GetPinID(resource.handle, ResourcePinType::TextureWrite), GetPinID(&resource.handle, PassPinType::PassTexture), IM_COL32(125, 0, 0, 255)));
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
							m_edges.emplace_back(std::make_tuple(GetPinID(resource.handle, ResourcePinType::BufferRead), GetPinID(&resource.handle, PassPinType::PassBuffer), IM_COL32(125, 0, 0, 255)));
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
							m_edges.emplace_back(std::make_tuple(GetPinID(resource.handle, ResourcePinType::BufferWrite), GetPinID(&resource.handle, PassPinType::PassBuffer), IM_COL32(125, 0, 0, 255)));
						}
					}
					ImNodes::PopColorStyle();
					ImNodes::PopColorStyle();

					if (resource.handle.IsValid())
					{
						m_edges.emplace_back(std::make_tuple(GetPinID(resource.handle, ResourcePinType::BufferRead), GetPinID(&resource.handle, PassPinType::PassBuffer), IM_COL32(125, 0, 0, 255)));
					}
				}
			}

			ImNodes::EndNode();
		}

		ImNodes::PopColorStyle();
	}

	void DrawEdges()
	{
		for (auto &[src, dst, color] : m_edges)
		{
			ImNodes::PushColorStyle(ImNodesCol_Link, color);
			ImNodes::Link(static_cast<int32_t>(Hash(src, dst)), src, dst);
			ImNodes::PopColorStyle();
		}
	}

	void AddEdge()
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

	void EditTextureNode()
	{
		for (auto &[handle, texture] : m_desc.textures)
		{
			if (ImNodes::IsNodeSelected(static_cast<int32_t>(handle.GetHandle())))
			{
				ImGui::PushID(static_cast<int32_t>(handle.GetHandle()));
				{
					char buf[64] = {0};
					std::memcpy(buf, texture.name.data(), sizeof(buf));
					if (ImGui::InputText("Texture", buf, sizeof(buf)))
					{
						texture.name = buf;
					}
				}
				uint32_t min_value = 1;
				uint32_t max_value = 4096;
				ImGui::DragScalar("Width", ImGuiDataType_U32, &texture.width, 1.f, &min_value, &max_value);
				ImGui::DragScalar("Height", ImGuiDataType_U32, &texture.height, 1.f, &min_value, &max_value);
				ImGui::DragScalar("Depth", ImGuiDataType_U32, &texture.depth, 1.f, &min_value, &max_value);
				ImGui::DragScalar("Layer", ImGuiDataType_U32, &texture.layers, 1.f, &min_value, &max_value);
				const char *const formats[] = {
				    "Undefined",
				    "R16_UINT",
				    "R16_SINT",
				    "R16_FLOAT",
				    "R8G8B8A8_UNORM",
				    "B8G8R8A8_UNORM",
				    "R32_UINT",
				    "R32_SINT",
				    "R32_FLOAT",
				    "D32_FLOAT",
				    "D24_UNORM_S8_UINT",
				    "R16G16_UINT",
				    "R16G16_SINT",
				    "R16G16_FLOAT",
				    "R10G10B10A2_UNORM",
				    "R10G10B10A2_UINT",
				    "R11G11B10_FLOAT",
				    "R16G16B16A16_UINT",
				    "R16G16B16A16_SINT",
				    "R16G16B16A16_FLOAT",
				    "R32G32_UINT",
				    "R32G32_SINT",
				    "R32G32_FLOAT",
				    "R32G32B32_UINT",
				    "R32G32B32_SINT",
				    "R32G32B32_FLOAT",
				    "R32G32B32A32_UINT",
				    "R32G32B32A32_SINT",
				    "R32G32B32A32_FLOAT"};
				ImGui::Combo("Format", reinterpret_cast<int *>(&texture.format), formats, 29);
				ImGui::PopID();
				ImGui::Separator();
			}
		}
	}

	void EditBufferNode()
	{
		for (auto &[handle, buffer] : m_desc.buffers)
		{
			if (ImNodes::IsNodeSelected(static_cast<int32_t>(handle.GetHandle())))
			{
				ImGui::PushID(static_cast<int32_t>(handle.GetHandle()));
				{
					char buf[64] = {0};
					std::memcpy(buf, buffer.name.data(), sizeof(buf));
					if (ImGui::InputText("Buffer", buf, sizeof(buf)))
					{
						buffer.name = buf;
					}
				}
				uint32_t min_value = 1;
				uint32_t max_value = 4096 * 4096;
				ImGui::DragScalar("Size", ImGuiDataType_U32, &buffer.size, 1.f, &min_value, &max_value);
				ImGui::PopID();
				ImGui::Separator();
			}
		}
	}

	void EditPassNode()
	{
		for (auto &[handle, pass] : m_desc.passes)
		{
			if (ImNodes::IsNodeSelected(static_cast<int32_t>(handle.GetHandle())))
			{
				const std::unordered_map<BindPoint, const char *> bind_points =
				    {
				        {BindPoint::None, "None"},
				        {BindPoint::Rasterization, "Rasterization"},
				        {BindPoint::Compute, "Compute"},
				        {BindPoint::RayTracing, "RayTracing"},
				        {BindPoint::CUDA, "CUDA"},
				    };

				ImGui::PushID(static_cast<int32_t>(handle.GetHandle()));
				ImGui::Text("Pass - %s", pass.name.c_str());
				ImGui::Text("Bind Point: %s", bind_points.at(pass.bind_point));
				PluginManager::GetInstance().Call<bool>(fmt::format("Pass.{}.dll", pass.name), "OnImGui", &pass.config, ImGui::GetCurrentContext());
				ImGui::PopID();
				ImGui::Separator();
			}
		}
	}

	void DrawInspector()
	{
		ImGui::BeginChild("Render Graph Inspector", ImVec2(0, 0), false, ImGuiWindowFlags_HorizontalScrollbar);

		ImGui::PushItemWidth(ImGui::GetColumnWidth(0) * 0.7f);

		EditTextureNode();
		EditBufferNode();
		EditPassNode();

		ImGui::PopItemWidth();

		if (ImGui::GetScrollY() >= ImGui::GetScrollMaxY())
		{
			ImGui::SetScrollHereY(1.0f);
		}

		ImGui::EndChild();
	}

  private:
	int32_t GetPinID(const RGHandle &handle, ResourcePinType pin)
	{
		int32_t id         = static_cast<int32_t>(Hash(handle.GetHandle(), pin));
		m_resource_pin[id] = std::make_pair(handle, pin);
		return id;
	}

	int32_t GetPinID(RGHandle *handle, PassPinType pin)
	{
		int32_t id     = static_cast<int32_t>(Hash(handle, pin));
		m_pass_pin[id] = std::make_pair(handle, pin);
		return id;
	}

	bool ValidLink(ResourcePinType resource, PassPinType pass)
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

  private:
	ImNodesEditorContext *m_context = nullptr;

	RenderGraphDesc m_desc;

	size_t m_current_handle = 0;

	std::unordered_map<int32_t, std::pair<RGHandle *, PassPinType>>   m_pass_pin;
	std::unordered_map<int32_t, std::pair<RGHandle, ResourcePinType>> m_resource_pin;

	std::vector<int32_t> m_select_nodes;
	std::vector<int32_t> m_select_links;

	std::vector<std::tuple<int32_t, int32_t, uint32_t>> m_edges;
};

extern "C"
{
	EXPORT_API RenderGraphEditor *Create(Editor *editor, ImGuiContext *context)
	{
		ImGui::SetCurrentContext(context);
		return new RenderGraphEditor(editor);
	}
}