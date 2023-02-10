#include <Editor/Editor.hpp>
#include <Editor/Widget.hpp>
#include <Material/MaterialData.hpp>
#include <Material/MaterialGraph.hpp>
#include <RenderGraph/RenderGraphBlackboard.hpp>
#include <Renderer/RenderData.hpp>
#include <Renderer/Renderer.hpp>
#include <Resource/Resource/Material.hpp>
#include <Resource/Resource/Mesh.hpp>
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

		std::vector<Resource<ResourceType::Mesh>::Vertex> vertices;

		std::vector<uint32_t> indices;

		DESERIALIZE("Asset/BuildIn/MaterialBall.asset", vertices, indices);

		auto *rhi_context = editor->GetRenderer()->GetRHIContext();
		auto *renderer    = editor->GetRenderer();

		m_view.buffer = rhi_context->CreateBuffer(sizeof(m_view.uniform_block), RHIBufferUsage::ConstantBuffer, RHIMemoryUsage::CPU_TO_GPU);
		m_view.Reset();
		m_view.Update();

		{
			m_view.uniform_block.model = glm::mat4_cast(glm::qua<float>(glm::radians(glm::vec3(135.f, 0.f, 180.f))));
		}

		m_material_ball.vertex_buffer = rhi_context->CreateBuffer<Resource<ResourceType::Mesh>::Vertex>(vertices.size(), RHIBufferUsage::Vertex | RHIBufferUsage::UnorderedAccess | RHIBufferUsage::Transfer, RHIMemoryUsage::GPU_Only);
		m_material_ball.index_buffer  = rhi_context->CreateBuffer<uint32_t>(indices.size(), RHIBufferUsage::Index | RHIBufferUsage::UnorderedAccess | RHIBufferUsage::Transfer, RHIMemoryUsage::GPU_Only);

		m_material_ball.indices_count = static_cast<uint32_t>(indices.size());
		m_material_ball.vertex_buffer->CopyToDevice(vertices.data(), vertices.size() * sizeof(Resource<ResourceType::Mesh>::Vertex));
		m_material_ball.index_buffer->CopyToDevice(indices.data(), indices.size() * sizeof(uint32_t));

		m_preview.pipeline = rhi_context->CreatePipelineState();

		BlendState blend_state;
		blend_state.attachment_states.resize(1);
		m_preview.pipeline->SetBlendState(blend_state);

		RasterizationState rasterization_state;
		rasterization_state.cull_mode  = RHICullMode::None;
		rasterization_state.front_face = RHIFrontFace::Clockwise;
		m_preview.pipeline->SetRasterizationState(rasterization_state);

		DepthStencilState depth_stencil_state  = {};
		depth_stencil_state.depth_write_enable = true;
		depth_stencil_state.depth_test_enable  = true;
		m_preview.pipeline->SetDepthStencilState(depth_stencil_state);

		VertexInputState vertex_input_state = {};
		vertex_input_state.input_bindings   = {
            VertexInputState::InputBinding{0, sizeof(Resource<ResourceType::Mesh>::Vertex), RHIVertexInputRate::Vertex}};
		vertex_input_state.input_attributes = {
		    VertexInputState::InputAttribute{RHIVertexSemantics::Position, 0, 0, RHIFormat::R32G32B32_FLOAT, offsetof(Resource<ResourceType::Mesh>::Vertex, position)},
		    VertexInputState::InputAttribute{RHIVertexSemantics::Normal, 1, 0, RHIFormat::R32G32B32_FLOAT, offsetof(Resource<ResourceType::Mesh>::Vertex, normal)},
		    VertexInputState::InputAttribute{RHIVertexSemantics::Texcoord, 3, 0, RHIFormat::R32G32_FLOAT, offsetof(Resource<ResourceType::Mesh>::Vertex, texcoord0)},
		};
		m_preview.pipeline->SetVertexInputState(vertex_input_state);

		m_preview.render_target_texture = rhi_context->CreateTexture2D(500, 500, RHIFormat::R8G8B8A8_UNORM, RHITextureUsage::RenderTarget | RHITextureUsage::ShaderResource, false);
		m_preview.depth_stencil_texture = rhi_context->CreateTexture2D(500, 500, RHIFormat::D32_FLOAT, RHITextureUsage::RenderTarget | RHITextureUsage::ShaderResource, false);
		m_preview.render_target         = rhi_context->CreateRenderTarget();
		m_preview.render_target->Set(0, m_preview.render_target_texture.get(), RHITextureDimension::Texture2D, ColorAttachment{});
		m_preview.render_target->Set(m_preview.depth_stencil_texture.get(), RHITextureDimension::Texture2D, DepthStencilAttachment{});

		{
			auto *cmd_buffer = rhi_context->CreateCommand(RHIQueueFamily::Graphics);
			cmd_buffer->Begin();
			cmd_buffer->ResourceStateTransition({TextureStateTransition{m_preview.render_target_texture.get(), RHIResourceState::Undefined, RHIResourceState::ShaderResource, TextureRange{}}}, {});
			cmd_buffer->End();
			rhi_context->Execute({cmd_buffer});
		}
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

		auto *resource = resource_manager->Get<ResourceType::Material>(m_material_name);

		ImGui::Columns(2);

		// Inspector
		{
			ImGui::BeginChild("Material Editor Inspector", ImVec2(0, 0), false, ImGuiWindowFlags_HorizontalScrollbar);

			ImGui::Text("Material Editor Inspector");

			if (resource)
			{
				Render(resource);
				float width = glm::min(ImGui::GetColumnWidth(), 300.f);
				ImGui::Image(m_preview.render_target_texture.get(), ImVec2(width, width));
				UpdateCamera();
			}

			SetMaterial(resource, resource_manager);

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

		ImGui::End();
	}

	void SetMaterial(Resource<ResourceType::Material> *resource, ResourceManager *manager)
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
				manager->Add<ResourceType::Material>(p_editor->GetRHIContext(), m_material_name, std::move(desc));
				resource = manager->Get<ResourceType::Material>(m_material_name);
			}
		}
		else
		{
			ImGui::Text("Material Name");
		}
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
					m_view.uniform_block.material_id = static_cast<uint32_t>(p_editor->GetRenderer()->GetResourceManager()->Index<ResourceType::Material>(m_material_name));
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
				DummyTexture *dummy_texture = p_editor->GetRenderer()->GetRenderGraphBlackboard().Get<DummyTexture>();
				GPUScene     *gpu_scene     = p_editor->GetRenderer()->GetRenderGraphBlackboard().Get<GPUScene>();

				resource->Compile(
				    p_editor->GetRenderer()->GetRHIContext(),
				    p_editor->GetRenderer()->GetResourceManager(),
				    dummy_texture->black_opaque.get(),
				    ImNodes::SaveCurrentEditorStateToIniString());
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
				ImGui::DragFloat("", pin.variant.Convert<float>(), 0.001f, 0.f, 0.f, "%.2f");
				break;
			case MaterialNodePin::Type::Float3:
				ImGui::DragFloat3("", glm::value_ptr(*pin.variant.Convert<glm::vec3>()), 0.001f, 0.f, 0.f, "%.2f");
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
		for (auto &[node_handle, node_desc] : resource->GetDesc().GetNodes())
		{
			const float node_width = glm::max(ImGui::CalcTextSize(node_desc.GetName().c_str()).x, 120.f);
			ImGui::PushItemWidth(node_width);

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
			ImGui::PopItemWidth();
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

	void DrawEdges(Resource<ResourceType::Material> *resource)
	{
		for (auto &[dst, src] : resource->GetDesc().GetEdges())
		{
			const auto &src_node = resource->GetDesc().GetNode(src);
			const auto &dst_node = resource->GetDesc().GetNode(dst);

			auto type = src_node.GetPin(src).type;

			ImNodes::PushColorStyle(ImNodesCol_Link, m_pin_color[type]);
			ImNodes::Link(static_cast<int32_t>(Hash(src, dst)), static_cast<int32_t>(src), static_cast<int32_t>(dst));
			ImNodes::PopColorStyle();
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

	void UpdateCamera()
	{
		float delta_time = ImGui::GetIO().DeltaTime;

		if (ImGui::IsItemHovered())
		{
			if (ImGui::IsMouseDragging(ImGuiMouseButton_Right))
			{
				ImVec2 delta = ImGui::GetMouseDragDelta(ImGuiMouseButton_Right);
				ImGui::ResetMouseDragDelta(ImGuiMouseButton_Right);
				m_view.phi -= delta.y * delta_time * 50.f;
				m_view.theta -= delta.x * delta_time * 50.f;
			}
			m_view.radius += ImGui::GetIO().MouseWheel * delta_time * 10.f;
		}

		m_view.Update();
	}

	void Render(Resource<ResourceType::Material> *resource)
	{
		auto *renderer    = p_editor->GetRenderer();
		auto *rhi_context = p_editor->GetRHIContext();
		auto *gpu_scene   = renderer->GetRenderGraphBlackboard().Get<GPUScene>();

		auto &material_data = resource->GetMaterialData();

		ShaderMeta meta;

		{
			auto *vertex_shader = renderer->RequireShader("Source/Shaders/MaterialEditor.hlsl", "VSmain", RHIShaderStage::Vertex, {"USE_MATERIAL", material_data.signature}, {material_data.shader});
			auto *frag_shader   = renderer->RequireShader("Source/Shaders/MaterialEditor.hlsl", "PSmain", RHIShaderStage::Fragment, {"USE_MATERIAL", material_data.signature}, {material_data.shader});

			m_preview.pipeline->ClearShader();
			m_preview.pipeline->SetShader(RHIShaderStage::Vertex, vertex_shader);
			m_preview.pipeline->SetShader(RHIShaderStage::Fragment, frag_shader);

			meta = renderer->RequireShaderMeta(vertex_shader);
			meta += renderer->RequireShaderMeta(frag_shader);
		}

		auto *descriptor = rhi_context->CreateDescriptor(meta);
		descriptor->BindBuffer("UniformBuffer", m_view.buffer.get())
		    .BindTexture("Textures", gpu_scene->textures.texture_2d, RHITextureDimension::Texture2D)
		    .BindSampler("Samplers", gpu_scene->samplers)
		    .BindBuffer("MaterialOffsets", gpu_scene->material.material_offset.get())
		    .BindBuffer("MaterialBuffer", gpu_scene->material.material_buffer.get());

		auto *cmd_buffer = rhi_context->CreateCommand(RHIQueueFamily::Graphics);
		cmd_buffer->Begin();
		cmd_buffer->ResourceStateTransition({TextureStateTransition{m_preview.render_target_texture.get(), RHIResourceState::ShaderResource, RHIResourceState::RenderTarget, TextureRange{}}}, {});
		cmd_buffer->BeginRenderPass(m_preview.render_target.get());
		cmd_buffer->SetViewport(static_cast<float>(m_preview.render_target->GetWidth()), static_cast<float>(m_preview.render_target->GetHeight()));
		cmd_buffer->SetScissor(m_preview.render_target->GetWidth(), m_preview.render_target->GetHeight());
		cmd_buffer->BindDescriptor(descriptor);
		cmd_buffer->BindPipelineState(m_preview.pipeline.get());
		cmd_buffer->BindVertexBuffer(0, m_material_ball.vertex_buffer.get());
		cmd_buffer->BindIndexBuffer(m_material_ball.index_buffer.get());
		cmd_buffer->DrawIndexed(m_material_ball.indices_count);
		cmd_buffer->EndRenderPass();
		cmd_buffer->ResourceStateTransition({TextureStateTransition{m_preview.render_target_texture.get(), RHIResourceState::RenderTarget, RHIResourceState::ShaderResource, TextureRange{}}}, {});
		cmd_buffer->End();

		rhi_context->Submit({cmd_buffer});
	}

  private:
	ImNodesEditorContext *m_context = nullptr;

	std::string m_material_name = "";

	size_t m_current_handle = 0;

	std::vector<int32_t> m_select_nodes;
	std::vector<int32_t> m_select_links;
	std::vector<int32_t> m_new_nodes;

	struct
	{
		glm::vec3 center = glm::vec3(0.f);

		float radius = 4.f;
		float theta  = 0.f;
		float phi    = 60.f;

		void Reset()
		{
			center = glm::vec3(0.f);
			radius = 4.f;
			theta  = 0.f;
			phi    = 60.f;

			Update();
		}

		void Update()
		{
			glm::vec3 position  = center + radius * glm::vec3(glm::sin(glm::radians(phi)) * glm::sin(glm::radians(theta)), glm::cos(glm::radians(phi)), glm::sin(glm::radians(phi)) * glm::cos(glm::radians(theta)));
			glm::vec3 direction = glm::normalize(center - position);
			glm::vec3 right     = glm::normalize(glm::cross(direction, glm::vec3{0.f, 1.f, 0.f}));
			glm::vec3 up        = glm::normalize(glm::cross(right, direction));
			glm::mat4 transform = glm::perspective(glm::radians(45.f), 1.f, 0.01f, 1000.f) * glm::lookAt(position, center, up);

			uniform_block.transform  = transform;
			uniform_block.camera_pos = position;
			buffer->CopyToDevice(&uniform_block, sizeof(uniform_block));
		}

		struct
		{
			glm::mat4 transform;
			glm::mat4 model;
			glm::vec3 camera_pos;
			uint32_t  material_id;
		} uniform_block;

		std::unique_ptr<RHIBuffer> buffer = nullptr;
	} m_view;

	struct
	{
		std::unique_ptr<RHIBuffer> vertex_buffer = nullptr;
		std::unique_ptr<RHIBuffer> index_buffer  = nullptr;

		uint32_t indices_count = 0;
	} m_material_ball;

	struct
	{
		std::unique_ptr<RHIPipelineState> pipeline              = nullptr;
		std::unique_ptr<RHITexture>       render_target_texture = nullptr;
		std::unique_ptr<RHITexture>       depth_stencil_texture = nullptr;
		std::unique_ptr<RHIRenderTarget>  render_target         = nullptr;
	} m_preview;

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