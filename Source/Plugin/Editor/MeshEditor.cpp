
#include <Editor/Editor.hpp>
#include <Editor/Widget.hpp>
#include <Geometry/Mesh/Mesh.hpp>
#include <Geometry/MeshProcess.hpp>
#include <Renderer/Renderer.hpp>
#include <Resource/Importer.hpp>
#include <Resource/Resource/Mesh.hpp>
#include <Resource/ResourceManager.hpp>

#include <imgui.h>
#include <imgui_internal.h>

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

using namespace Ilum;

class MeshEditor : public Widget
{
  private:
	struct UniformBlock
	{
		glm::mat4 transform;
		alignas(16) glm::vec3 color;
		alignas(16) glm::vec3 direction;
	};

	enum class ShadingMode
	{
		None,
		Shading,
		Normal,
		UV,
		Texture
	};

  public:
	MeshEditor(Editor *editor) :
	    Widget("Mesh Editor", editor)
	{
		m_pipeline_state = p_editor->GetRHIContext()->CreatePipelineState();
		m_render_target  = p_editor->GetRHIContext()->CreateRenderTarget();
		m_uniform_buffer = p_editor->GetRHIContext()->CreateBuffer<UniformBlock>(1, RHIBufferUsage::ConstantBuffer, RHIMemoryUsage::CPU_TO_GPU);

		VertexInputState vertex_input_state = {};
		vertex_input_state.input_bindings   = {
            VertexInputState::InputBinding{0, sizeof(Resource<ResourceType::Mesh>::Vertex), RHIVertexInputRate::Vertex}};

		BlendState blend_state = {};
		blend_state.attachment_states.resize(1);

		DepthStencilState depth_stencil_state  = {};
		depth_stencil_state.depth_test_enable  = true;
		depth_stencil_state.depth_write_enable = true;

		m_pipeline_state->SetVertexInputState(vertex_input_state);
		m_pipeline_state->SetBlendState(blend_state);
		m_pipeline_state->SetDepthStencilState(depth_stencil_state);
	}

	virtual ~MeshEditor() override = default;

	virtual void Tick() override
	{
		if (!ImGui::Begin(m_name.c_str()))
		{
			ImGui::End();
			return;
		}

		ImGui::Columns(2);

		ImGui::SetColumnWidth(0, ImGui::GetWindowWidth() * 0.2f);

		// Inspector
		{
			ImGui::BeginChild("Render Graph Inspector", ImVec2(0, 0), false, ImGuiWindowFlags_HorizontalScrollbar);

			ImGui::Text("Mesh Editor Inspector");

			if (ImGui::TreeNode("Mesh"))
			{
				ImGui::Text("Mesh Name: %s", m_mesh_name.c_str());
				ImGui::Text("Vertices Count: %ld", m_mesh.vertices.size());
				ImGui::Text("Triangle Count: %ld", m_mesh.indices.size() / 3);

				if (ImGui::Button(m_mesh_name.c_str(), ImVec2(150.f, 20.f)))
				{
					m_mesh_name = "";
					m_mesh.vertices.clear();
					m_mesh.indices.clear();
				}

				if (ImGui::BeginDragDropTarget())
				{
					if (const auto *pay_load = ImGui::AcceptDragDropPayload("Mesh"))
					{
						m_mesh_name    = static_cast<const char *>(pay_load->Data);
						auto *resource = p_editor->GetRenderer()->GetResourceManager()->Get<ResourceType::Mesh>(m_mesh_name);
						m_mesh.indices = resource->GetIndices();
						m_mesh.vertices.resize(resource->GetVertices().size());

						glm::vec3 min_bound = glm::vec3(std::numeric_limits<float>::max());
						glm::vec3 max_bound = glm::vec3(-std::numeric_limits<float>::max());

						for (uint32_t i = 0; i < m_mesh.vertices.size(); i++)
						{
							VertexData data = {};
							data.position   = resource->GetVertices()[i].position;
							data.normal     = resource->GetVertices()[i].normal;
							data.uv         = resource->GetVertices()[i].texcoord0;

							min_bound = glm::min(min_bound, data.position);
							max_bound = glm::max(max_bound, data.position);

							m_mesh.vertices[i] = data;
						}

						m_view.center /= static_cast<float>(m_mesh.vertices.size());
						m_view.radius  = glm::length(max_bound - min_bound);
						m_scale_factor = m_view.radius;
					}
					ImGui::EndDragDropTarget();
				}

				ImGui::TreePop();
			}

			if (ImGui::TreeNode("Rendering"))
			{
				ImGui::Checkbox("Wireframe", &m_wireframe);

				const char *shading_mode[] = {"None", "Shading", "Normal", "UV", "Texture"};
				ImGui::PushItemWidth(90);
				ImGui::Combo("Mode", reinterpret_cast<int32_t *>(&m_shading_mode), shading_mode, 5);
				ImGui::PopItemWidth();
				if (m_shading_mode == ShadingMode::Shading)
				{
					ImGui::ColorEdit3("Color", glm::value_ptr(m_color));
				}
				ImGui::ColorEdit3("Background Color", glm::value_ptr(m_bg_color));
				ImGui::TreePop();
			}

			if (ImGui::TreeNode("Process"))
			{
				for (const auto &file : std::filesystem::directory_iterator("./lib/"))
				{
					std::string filename = file.path().filename().string();
					if (std::regex_match(filename, std::regex("(Geometry.)Subdivision.(.*)(.dll)")))
					{
						if (ImGui::TreeNode("Subdivision"))
						{
							size_t begin = std::string("Geometry.Subdivision.").length();
							size_t end   = filename.find_first_of('.', begin);

							if (ImGui::Button(filename.substr(begin, end - begin).c_str()))
							{
								m_mesh = Subdivision::GetInstance(filename)->Execute(m_mesh);
								UpdateBuffer();
							}
							ImGui::TreePop();
						}
					}
				}

				ImGui::TreePop();
			}

			if (ImGui::GetScrollY() >= ImGui::GetScrollMaxY())
			{
				ImGui::SetScrollHereY(1.0f);
			}

			ImGui::EndChild();
		}

		ImGui::NextColumn();

		UpdateCamera(ImGui::GetColumnWidth(1), ImGui::GetContentRegionAvail().y);

		// Model Viewer
		if (p_editor->GetRenderer()->GetResourceManager()->Has<ResourceType::Mesh>(m_mesh_name))
		{
			Render(static_cast<uint32_t>(ImGui::GetColumnWidth(1)), static_cast<uint32_t>(ImGui::GetContentRegionAvail().y));
			if (m_render_texture)
			{
				ImGui::Image(m_render_texture.get(), ImVec2{ImGui::GetColumnWidth(1), ImGui::GetContentRegionAvail().y});
			}
		}

		ImGui::End();
	}

	void UpdateBuffer()
	{
		auto *rhi_context      = p_editor->GetRenderer()->GetRHIContext();
		auto *resource_manager = p_editor->GetRenderer()->GetResourceManager();

		if (resource_manager->Has<ResourceType::Mesh>(m_mesh_name))
		{
			auto *resource = resource_manager->Get<ResourceType::Mesh>(m_mesh_name);

			std::vector<Resource<ResourceType::Mesh>::Vertex> vertices(m_mesh.vertices.size());
			std::transform(m_mesh.vertices.begin(), m_mesh.vertices.end(), vertices.begin(), [](const auto &data) {
				Resource<ResourceType::Mesh>::Vertex v = {};

				v.position  = data.position;
				v.normal    = data.normal;
				v.texcoord0 = data.uv;

				return v;
			});
			std::vector<uint32_t> indices = m_mesh.indices;

			resource->Update(rhi_context, std::move(vertices), std::move(indices));
		}
	}

	void Render(uint32_t width, uint32_t height)
	{
		auto *rhi_context = p_editor->GetRHIContext();
		auto *resource    = p_editor->GetRenderer()->GetResourceManager()->Get<ResourceType::Mesh>(m_mesh_name);

		auto *cmd_buffer = rhi_context->CreateCommand(RHIQueueFamily::Graphics);
		cmd_buffer->Begin();

		if (!m_uv_texture)
		{
			m_uv_texture = rhi_context->CreateTexture2D(500, 500, RHIFormat::R8G8B8A8_UNORM, RHITextureUsage::ShaderResource | RHITextureUsage::Transfer, false);
			std::vector<uint8_t> data(500 * 500 * 4);
			for (size_t i = 0; i < 500; i++)
			{
				for (size_t j = 0; j < 500; j++)
				{
					if (((i / 50) % 2 == 0 && (j / 50) % 2 == 1) ||
					    ((i / 50) % 2 == 1 && (j / 50) % 2 == 0))
					{
						data[500U * 4U * i + 4U * j]     = 0;
						data[500U * 4U * i + 4U * j + 1] = 125;
						data[500U * 4U * i + 4U * j + 2] = 0;
						data[500U * 4U * i + 4U * j + 3] = 255;
					}
					else
					{
						data[500U * 4U * i + 4U * j]     = 255;
						data[500U * 4U * i + 4U * j + 1] = 255;
						data[500U * 4U * i + 4U * j + 2] = 255;
						data[500U * 4U * i + 4U * j + 3] = 255;
					}
				}
			}

			{
				auto *copy_cmd_buffer = rhi_context->CreateCommand(RHIQueueFamily::Graphics);
				copy_cmd_buffer->Begin();
				copy_cmd_buffer->ResourceStateTransition({TextureStateTransition{m_uv_texture.get(), RHIResourceState::Undefined, RHIResourceState::TransferDest, TextureRange{RHITextureDimension::Texture2D, 0, 1, 0, 1}}}, {});
				auto staging_buffer = rhi_context->CreateBuffer(data.size(), RHIBufferUsage::Transfer, RHIMemoryUsage::CPU_TO_GPU);
				staging_buffer->CopyToDevice(data.data(), data.size());
				copy_cmd_buffer->CopyBufferToTexture(staging_buffer.get(), m_uv_texture.get(), 0, 0, 1);
				copy_cmd_buffer->End();
				rhi_context->Execute(copy_cmd_buffer);
			}

			cmd_buffer->ResourceStateTransition({TextureStateTransition{m_uv_texture.get(), RHIResourceState::TransferDest, RHIResourceState::ShaderResource, TextureRange{RHITextureDimension::Texture2D, 0, 1, 0, 1}}}, {});
		}

		if (!m_render_texture)
		{
			m_render_texture = rhi_context->CreateTexture2D(1000, 1000, RHIFormat::R8G8B8A8_UNORM, RHITextureUsage::RenderTarget | RHITextureUsage::ShaderResource, false);
			m_depth_texture  = rhi_context->CreateTexture2D(1000, 1000, RHIFormat::D32_FLOAT, RHITextureUsage::RenderTarget, false);
			cmd_buffer->ResourceStateTransition(
			    {TextureStateTransition{m_render_texture.get(), RHIResourceState::Undefined, RHIResourceState::RenderTarget, TextureRange{RHITextureDimension::Texture2D, 0, 1, 0, 1}},
			     TextureStateTransition{m_depth_texture.get(), RHIResourceState::Undefined, RHIResourceState::DepthWrite, TextureRange{RHITextureDimension::Texture2D, 0, 1, 0, 1}}},
			    {});
		}
		else
		{
			cmd_buffer->ResourceStateTransition({TextureStateTransition{m_render_texture.get(), RHIResourceState::ShaderResource, RHIResourceState::RenderTarget, TextureRange{RHITextureDimension::Texture2D, 0, 1, 0, 1}}}, {});
		}

		// Shading
		if (m_shading_mode != ShadingMode::None)
		{
			const char *shading_macros[] = {"NONE", "SHADING", "DRAW_NORMAL", "DRAW_UV", "DRAW_TEXTURE"};

			auto *vertex_shader   = p_editor->GetRenderer()->RequireShader("./Source/Shaders/MeshEditor.hlsl", "VSmain", RHIShaderStage::Vertex, {shading_macros[(size_t) m_shading_mode]});
			auto *fragment_shader = p_editor->GetRenderer()->RequireShader("./Source/Shaders/MeshEditor.hlsl", "PSmain", RHIShaderStage::Fragment, {shading_macros[(size_t) m_shading_mode]});

			m_pipeline_state->ClearShader();
			m_pipeline_state->SetShader(RHIShaderStage::Vertex, vertex_shader);
			m_pipeline_state->SetShader(RHIShaderStage::Fragment, fragment_shader);

			ShaderMeta meta = p_editor->GetRenderer()->RequireShaderMeta(vertex_shader);
			meta += p_editor->GetRenderer()->RequireShaderMeta(fragment_shader);

			RasterizationState rasterization_state = {};
			rasterization_state.polygon_mode       = RHIPolygonMode::Solid;
			rasterization_state.cull_mode          = RHICullMode::None;
			m_pipeline_state->SetRasterizationState(rasterization_state);

			VertexInputState vertex_input_state = m_pipeline_state->GetVertexInputState();
			if (m_shading_mode == ShadingMode::Shading ||
			    m_shading_mode == ShadingMode::Normal)
			{
				vertex_input_state.input_attributes = {
				    VertexInputState::InputAttribute{RHIVertexSemantics::Position, 0, 0, RHIFormat::R32G32B32_FLOAT, offsetof(Resource<ResourceType::Mesh>::Vertex, position)},
				    VertexInputState::InputAttribute{RHIVertexSemantics::Normal, 1, 0, RHIFormat::R32G32B32_FLOAT, offsetof(Resource<ResourceType::Mesh>::Vertex, normal)},
				};
			}
			else if (m_shading_mode == ShadingMode::UV)
			{
				vertex_input_state.input_attributes = {
				    VertexInputState::InputAttribute{RHIVertexSemantics::Position, 0, 0, RHIFormat::R32G32B32_FLOAT, offsetof(Resource<ResourceType::Mesh>::Vertex, position)},
				    VertexInputState::InputAttribute{RHIVertexSemantics::Texcoord, 1, 0, RHIFormat::R32G32_FLOAT, offsetof(Resource<ResourceType::Mesh>::Vertex, texcoord0)},
				};
			}
			else if (m_shading_mode == ShadingMode::Texture)
			{
				vertex_input_state.input_attributes = {
				    VertexInputState::InputAttribute{RHIVertexSemantics::Position, 0, 0, RHIFormat::R32G32B32_FLOAT, offsetof(Resource<ResourceType::Mesh>::Vertex, position)},
				    VertexInputState::InputAttribute{RHIVertexSemantics::Normal, 1, 0, RHIFormat::R32G32B32_FLOAT, offsetof(Resource<ResourceType::Mesh>::Vertex, normal)},
				    VertexInputState::InputAttribute{RHIVertexSemantics::Texcoord, 2, 0, RHIFormat::R32G32_FLOAT, offsetof(Resource<ResourceType::Mesh>::Vertex, texcoord0)},
				};
			}
			m_pipeline_state->SetVertexInputState(vertex_input_state);

			m_render_target->Clear();
			m_render_target->Set(0, m_render_texture.get(), TextureRange{RHITextureDimension::Texture2D, 0, 1, 0, 1}, ColorAttachment{RHILoadAction::Clear, RHIStoreAction::Store, {m_bg_color.x, m_bg_color.y, m_bg_color.z, 1.f}});
			m_render_target->Set(m_depth_texture.get(), TextureRange{RHITextureDimension::Texture2D, 0, 1, 0, 1}, DepthStencilAttachment{});

			auto *descriptor = rhi_context->CreateDescriptor(meta);
			descriptor->BindTexture("UVTexture", m_uv_texture.get(), RHITextureDimension::Texture2D)
			    .BindSampler("UVSampler", rhi_context->CreateSampler(SamplerDesc::LinearWarp))
			    .BindBuffer("UniformBuffer", m_uniform_buffer.get());

			cmd_buffer->BeginRenderPass(m_render_target.get());
			cmd_buffer->BindDescriptor(descriptor);
			cmd_buffer->BindPipelineState(m_pipeline_state.get());
			cmd_buffer->BindVertexBuffer(0, resource->GetVertexBuffer());
			cmd_buffer->BindIndexBuffer(resource->GetIndexBuffer());
			cmd_buffer->SetViewport((float) m_render_target->GetWidth(), (float) m_render_target->GetHeight());
			cmd_buffer->SetScissor(m_render_target->GetWidth(), m_render_target->GetHeight());
			cmd_buffer->DrawIndexed(static_cast<uint32_t>(m_mesh.indices.size()));
			cmd_buffer->EndRenderPass();
		}

		if (m_wireframe)
		{
			auto *vertex_shader   = p_editor->GetRenderer()->RequireShader("./Source/Shaders/MeshEditor.hlsl", "VSmain", RHIShaderStage::Vertex, {"WIREFRAME"});
			auto *fragment_shader = p_editor->GetRenderer()->RequireShader("./Source/Shaders/MeshEditor.hlsl", "PSmain", RHIShaderStage::Fragment, {"WIREFRAME"});

			m_pipeline_state->ClearShader();
			m_pipeline_state->SetShader(RHIShaderStage::Vertex, vertex_shader);
			m_pipeline_state->SetShader(RHIShaderStage::Fragment, fragment_shader);

			ShaderMeta meta = p_editor->GetRenderer()->RequireShaderMeta(vertex_shader);
			meta += p_editor->GetRenderer()->RequireShaderMeta(fragment_shader);

			RasterizationState rasterization_state = {};
			rasterization_state.polygon_mode       = RHIPolygonMode::Wireframe;
			rasterization_state.cull_mode          = RHICullMode::None;
			m_pipeline_state->SetRasterizationState(rasterization_state);

			VertexInputState vertex_input_state = m_pipeline_state->GetVertexInputState();
			vertex_input_state.input_attributes = {
			    VertexInputState::InputAttribute{RHIVertexSemantics::Position, 0, 0, RHIFormat::R32G32B32_FLOAT, offsetof(Resource<ResourceType::Mesh>::Vertex, position)},
			};
			m_pipeline_state->SetVertexInputState(vertex_input_state);

			m_render_target->Clear();
			m_render_target->Set(0, m_render_texture.get(), TextureRange{RHITextureDimension::Texture2D, 0, 1, 0, 1}, ColorAttachment{m_shading_mode != ShadingMode::None ? RHILoadAction::Load : RHILoadAction::Clear, RHIStoreAction::Store, {0.2f, 0.3f, 0.3f, 1.f}});
			m_render_target->Set(m_depth_texture.get(), TextureRange{RHITextureDimension::Texture2D, 0, 1, 0, 1}, DepthStencilAttachment{m_shading_mode != ShadingMode::None ? RHILoadAction::Load : RHILoadAction::Clear});

			auto *descriptor = rhi_context->CreateDescriptor(meta);
			descriptor->BindTexture("UVTexture", m_uv_texture.get(), RHITextureDimension::Texture2D)
			    .BindSampler("UVSampler", rhi_context->CreateSampler(SamplerDesc::LinearWarp))
			    .BindBuffer("UniformBuffer", m_uniform_buffer.get());

			cmd_buffer->BeginRenderPass(m_render_target.get());
			cmd_buffer->BindDescriptor(descriptor);
			cmd_buffer->BindPipelineState(m_pipeline_state.get());
			cmd_buffer->BindVertexBuffer(0, resource->GetVertexBuffer());
			cmd_buffer->BindIndexBuffer(resource->GetIndexBuffer());
			cmd_buffer->SetViewport((float) m_render_target->GetWidth(), (float) m_render_target->GetHeight());
			cmd_buffer->SetScissor(m_render_target->GetWidth(), m_render_target->GetHeight());
			cmd_buffer->DrawIndexed(static_cast<uint32_t>(m_mesh.indices.size()));
			cmd_buffer->EndRenderPass();
		}

		cmd_buffer->ResourceStateTransition({TextureStateTransition{m_render_texture.get(), RHIResourceState::RenderTarget, RHIResourceState::ShaderResource, TextureRange{RHITextureDimension::Texture2D, 0, 1, 0, 1}}}, {});
		cmd_buffer->End();
		rhi_context->Submit({cmd_buffer});
	}

	void UpdateCamera(float width, float height)
	{
		float delta_time = ImGui::GetIO().DeltaTime;

		if (ImGui::IsWindowHovered())
		{
			if (ImGui::IsMouseDragging(ImGuiMouseButton_Right))
			{
				ImVec2 delta = ImGui::GetMouseDragDelta(ImGuiMouseButton_Right);
				ImGui::ResetMouseDragDelta(ImGuiMouseButton_Right);
				m_view.phi -= delta.y * delta_time * 5.f;
				m_view.theta -= delta.x * delta_time * 5.f;
			}
			m_view.radius += m_scale_factor * ImGui::GetIO().MouseWheel * delta_time;
		}

		// if (ImGui::IsMouseDragging(ImGuiMouseButton_Middle))
		//{
		//	ImVec2 delta = ImGui::GetMouseDragDelta(ImGuiMouseButton_Middle);
		//	ImGui::ResetMouseDragDelta(ImGuiMouseButton_Middle);
		//	m_view.radius += m_scale_factor * delta.y * delta_time;
		// }
		//  z=x, x = y, y= z
		glm::vec3 position = m_view.center + m_view.radius * glm::vec3(glm::sin(glm::radians(m_view.phi)) * glm::sin(glm::radians(m_view.theta)), glm::cos(glm::radians(m_view.phi)), glm::sin(glm::radians(m_view.phi)) * glm::cos(glm::radians(m_view.theta)));

		UniformBlock block = {};

		block.color     = m_color;
		block.direction = glm::normalize(m_view.center - position);
		glm::vec3 right = glm::normalize(glm::cross(block.direction, glm::vec3{0.f, 1.f, 0.f}));
		glm::vec3 up    = glm::normalize(glm::cross(right, block.direction));
		block.transform = glm::perspective(glm::radians(45.f), width / height, 0.01f, 1000.f) * glm::lookAt(position, m_view.center, up);

		m_uniform_buffer->CopyToDevice(&block, sizeof(block));
	}

  private:
	std::string m_mesh_name = "";

	TriMesh m_mesh;

	struct
	{
		glm::vec3 center = glm::vec3(0.f);
		float     radius = 0.f;
		float     theta  = 0.f;
		float     phi    = 90.f;
	} m_view;

	std::unique_ptr<RHIPipelineState> m_pipeline_state = nullptr;
	std::unique_ptr<RHIRenderTarget>  m_render_target  = nullptr;

	std::unique_ptr<RHIBuffer> m_uniform_buffer = nullptr;

	std::unique_ptr<RHITexture> m_uv_texture     = nullptr;
	std::unique_ptr<RHITexture> m_render_texture = nullptr;
	std::unique_ptr<RHITexture> m_depth_texture  = nullptr;

	glm::vec3 m_color    = glm::vec3(1.f);
	glm::vec3 m_bg_color = glm::vec3(0.2f, 0.3f, 0.3f);

	float m_scale_factor = 1.f;

	bool m_wireframe = false;

	ShadingMode m_shading_mode = ShadingMode::Shading;
};

extern "C"
{
	EXPORT_API MeshEditor *Create(Editor *editor, ImGuiContext *context)
	{
		ImGui::SetCurrentContext(context);
		return new MeshEditor(editor);
	}
}