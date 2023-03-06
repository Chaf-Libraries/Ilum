#include <Editor/Editor.hpp>
#include <Editor/Widget.hpp>
#include <Renderer/Renderer.hpp>
#include <Resource/Resource/Animation.hpp>
#include <Resource/Resource/SkinnedMesh.hpp>
#include <Resource/ResourceManager.hpp>

#include <imgui.h>
#include <imgui_internal.h>

using namespace Ilum;

class AnimationEditor : public Widget
{
  public:
	struct UniformBlock
	{
		glm::mat4 transform;
	};

	struct BoneVertex
	{
		alignas(16) glm::vec3 position;
		uint32_t bone_id;
	};

  public:
	AnimationEditor(Editor *editor) :
	    Widget("Animation Editor", editor)
	{
		m_uniform_buffer = p_editor->GetRHIContext()->CreateBuffer<UniformBlock>(1, RHIBufferUsage::ConstantBuffer, RHIMemoryUsage::CPU_TO_GPU);
		m_pipeline_state = p_editor->GetRHIContext()->CreatePipelineState();
		m_render_target  = p_editor->GetRHIContext()->CreateRenderTarget();

		VertexInputState vertex_input_state = {};
		vertex_input_state.input_bindings   = {
            VertexInputState::InputBinding{0, sizeof(BoneVertex), RHIVertexInputRate::Vertex}};
		vertex_input_state.input_attributes = {
		    VertexInputState::InputAttribute{RHIVertexSemantics::Blend_Indices, 0, 0, RHIFormat::R32_UINT, offsetof(BoneVertex, bone_id)},
		};

		BlendState blend_state = {};
		blend_state.attachment_states.resize(1);

		InputAssemblyState input_assembly_state = {};
		input_assembly_state.topology           = RHIPrimitiveTopology::Triangle;

		DepthStencilState depth_stencil_state  = {};
		depth_stencil_state.depth_test_enable  = true;
		depth_stencil_state.depth_write_enable = true;

		m_pipeline_state->SetVertexInputState(vertex_input_state);
		m_pipeline_state->SetBlendState(blend_state);
		m_pipeline_state->SetDepthStencilState(depth_stencil_state);
		m_pipeline_state->SetInputAssemblyState(input_assembly_state);
	}

	virtual ~AnimationEditor() override = default;

	virtual void Tick() override
	{
		if (!ImGui::Begin(m_name.c_str()))
		{
			ImGui::End();
			return;
		}

		auto *resource = p_editor->GetRenderer()->GetResourceManager()->Get<ResourceType::Animation>(m_animation_name);

		ImGui::Columns(2);

		// Inspector
		{
			ImGui::BeginChild("Animation Editor Inspector", ImVec2(0, 0), false, ImGuiWindowFlags_HorizontalScrollbar);

			ImGui::Text("Animation Editor Inspector");

			if (ImGui::TreeNode("Animation"))
			{
				if (ImGui::Button(m_animation_name.c_str(), ImVec2(150.f, 20.f)))
				{
					m_animation_name = "";
					m_root.children.clear();
					m_root.name = "";
				}

				if (ImGui::BeginDragDropTarget())
				{
					if (const auto *pay_load = ImGui::AcceptDragDropPayload("Animation"))
					{
						m_animation_name = static_cast<const char *>(pay_load->Data);
						resource         = p_editor->GetRenderer()->GetResourceManager()->Get<ResourceType::Animation>(m_animation_name);
						if (resource)
						{
							m_root = resource->GetHierarchyNode();

							glm::vec3 min_bound = glm::vec3(std::numeric_limits<float>::max());
							glm::vec3 max_bound = glm::vec3(-std::numeric_limits<float>::max());

							CalculateBound(resource, m_root, min_bound, max_bound);

							// Update skinned buffer
							UpdateBuffer(m_time);

							// Update vertex buffer
							std::vector<BoneVertex> bone_vertices;
							UpdateVertexBuffer(bone_vertices, m_root, resource);

							m_vertex_buffer = p_editor->GetRHIContext()->CreateBuffer<BoneVertex>(bone_vertices.size(), RHIBufferUsage::Vertex | RHIBufferUsage::Transfer, RHIMemoryUsage::GPU_Only);
							m_vertex_buffer->CopyToDevice(bone_vertices.data(), bone_vertices.size() * sizeof(BoneVertex));

							m_view.center  = 0.5f * (min_bound + max_bound);
							m_view.radius  = glm::length(max_bound - min_bound);
							m_scale_factor = m_view.radius;
						}
					}
					ImGui::EndDragDropTarget();
				}

				ImGui::SameLine();

				if (ImGui::Button(m_start ? "Stop" : "Start", ImVec2(150.f, 20.f)))
				{
					m_start = !m_start;
				}

				if (m_start)
				{
					m_time += ImGui::GetIO().DeltaTime;
					if (m_time > (resource ? resource->GetMaxTimeStamp() : 1000.f))
					{
						m_time -= resource ? resource->GetMaxTimeStamp() : 1000.f;
					}
					UpdateBuffer(m_time);
				}

				if (ImGui::SliderFloat("Time", &m_time, 0.f, resource ? resource->GetMaxTimeStamp() : 1000.f))
				{
					UpdateBuffer(m_time);
				}

				// Draw hierarchy
				if (!m_animation_name.empty())
				{
					DrawHierarchy(resource, m_root);
				}

				ImGui::TreePop();
			}

			if (ImGui::TreeNode("Skinned Meshes"))
			{
				for (auto &skinned_mesh : m_skinned_meshes)
				{
					ImGui::PushID(skinned_mesh.c_str());

					if (ImGui::Button(skinned_mesh.c_str(), ImVec2(ImGui::GetContentRegionAvail().x * 0.8f, 30.f)))
					{
						skinned_mesh = "";
					}

					if (ImGui::BeginDragDropTarget())
					{
						if (const auto *pay_load = ImGui::AcceptDragDropPayload("SkinnedMesh"))
						{
							skinned_mesh = static_cast<const char *>(pay_load->Data);
						}
					}

					ImGui::PopID();
				}

				if (ImGui::Button("+"))
				{
					m_skinned_meshes.emplace_back("");
				}

				ImGui::SameLine();

				if (ImGui::Button("-"))
				{
					m_skinned_meshes.pop_back();
				}
				ImGui::TreePop();
			}

			ImGui::EndChild();
		}

		ImGui::NextColumn();

		UpdateCamera(ImGui::GetColumnWidth(1), ImGui::GetContentRegionAvail().y);

		// Model Viewer
		if (resource)
		{
			Render(static_cast<uint32_t>(ImGui::GetColumnWidth(1)), static_cast<uint32_t>(ImGui::GetContentRegionAvail().y));
			if (m_render_texture)
			{
				ImGui::Image(m_render_texture.get(), ImVec2{ImGui::GetColumnWidth(1), ImGui::GetContentRegionAvail().y});
			}
		}

		ImGui::End();
	}

	void UpdateVertexBuffer(std::vector<BoneVertex> &vertices, const HierarchyNode &node, Resource<ResourceType::Animation> *resource, Bone *parent = nullptr, glm::mat4 parent_transform = glm::mat4(1.f))
	{
		auto *bone = resource->GetBone(node.name);

		if (bone)
		{
			uint32_t current_id = bone->GetBoneID();
			if (parent)
			{
				BoneVertex p0 = {}, p1 = {};
				p0.bone_id  = parent->GetBoneID();
				p1.bone_id  = current_id;
				p0.position = glm::vec3(parent_transform * glm::vec4(0.f, 0.f, 0.f, 1.f));
				p1.position = glm::vec3(parent_transform * node.transform * glm::vec4(0.f, 0.f, 0.f, 1.f));

				vertices.push_back(p0);
				vertices.push_back(p1);
			}
		}

		for (auto &child : node.children)
		{
			UpdateVertexBuffer(vertices, child, resource, bone, parent_transform * node.transform);
		}
	}

	void DrawHierarchy(Resource<ResourceType::Animation> *resource, const HierarchyNode &node)
	{
		Bone *bone = resource->GetBone(node.name);

		bool open = false;
		if (bone)
		{
			open = ImGui::TreeNodeEx(node.name.c_str(), node.children.empty() ? ImGuiTreeNodeFlags_Leaf : ImGuiTreeNodeFlags_None);
		}

		if (!bone || open)
		{
			for (auto &child : node.children)
			{
				DrawHierarchy(resource, child);
			}
		}

		if (open)
		{
			ImGui::TreePop();
		}
	}

	void CalculateBound(Resource<ResourceType::Animation> *resource, const HierarchyNode &node, glm::vec3 &min_bound, glm::vec3 &max_bound, glm::mat4 parent = glm::mat4(1.f))
	{
		Bone *bone = resource->GetBone(node.name);

		glm::mat4 global_transformation = parent * node.transform;

		if (bone)
		{
			global_transformation = parent * bone->GetLocalTransform(0.f);

			glm::vec3 position = glm::vec3(global_transformation * bone->GetBoneOffset() * glm::vec4(0.f, 0.f, 0.f, 1.f));

			min_bound = glm::min(min_bound, position);
			max_bound = glm::max(max_bound, position);
		}

		for (auto &child : node.children)
		{
			CalculateBound(resource, child, min_bound, max_bound, global_transformation);
		}
	}

	void CalculateBoneTransform(float time, Resource<ResourceType::Animation> *resource, const HierarchyNode &node, glm::mat4 *&skinned_matrices, glm::mat4 parent = glm::mat4(1.f))
	{
		Bone *bone = resource->GetBone(node.name);

		glm::mat4 global_transformation = parent * node.transform;

		if (bone)
		{
			global_transformation = parent * bone->GetLocalTransform(time);
			if (skinned_matrices)
			{
				uint32_t  bone_id = bone->GetBoneID();
				glm::mat4 offset  = bone->GetBoneOffset();

				skinned_matrices[bone_id] = global_transformation * offset;
			}
		}

		for (auto &child : node.children)
		{
			CalculateBoneTransform(time, resource, child, skinned_matrices, global_transformation);
		}
	}

	void UpdateBuffer(float time)
	{
		auto *resource    = p_editor->GetRenderer()->GetResourceManager()->Get<ResourceType::Animation>(m_animation_name);
		auto *rhi_context = p_editor->GetRHIContext();
		if (resource)
		{
			if (!m_skinned_buffer || m_skinned_buffer->GetDesc().size != resource->GetBoneCount() * sizeof(glm::mat4))
			{
				m_skinned_buffer = rhi_context->CreateBuffer<glm::mat4>((size_t) resource->GetMaxBoneIndex() + 1U, RHIBufferUsage::UnorderedAccess, RHIMemoryUsage::CPU_TO_GPU);
			}

			glm::mat4 *skinned_matrices = static_cast<glm::mat4 *>(m_skinned_buffer->Map());

			CalculateBoneTransform(time, resource, m_root, skinned_matrices);

			m_skinned_buffer->Unmap();
		}
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

		glm::vec3 position = m_view.center + m_view.radius * glm::vec3(glm::sin(glm::radians(m_view.phi)) * glm::sin(glm::radians(m_view.theta)), glm::cos(glm::radians(m_view.phi)), glm::sin(glm::radians(m_view.phi)) * glm::cos(glm::radians(m_view.theta)));

		UniformBlock block = {};

		glm::vec3 forward = glm::normalize(m_view.center - position);
		glm::vec3 right   = glm::normalize(glm::cross(forward, glm::vec3{0.f, 1.f, 0.f}));
		glm::vec3 up      = glm::normalize(glm::cross(right, forward));
		block.transform   = glm::perspective(glm::radians(45.f), width / height, 0.01f, 1000.f) * glm::lookAt(position, m_view.center, up);

		m_uniform_buffer->CopyToDevice(&block, sizeof(block));
	}

	void Render(uint32_t width, uint32_t height)
	{
		auto *rhi_context = p_editor->GetRHIContext();
		auto *resource    = p_editor->GetRenderer()->GetResourceManager()->Get<ResourceType::Animation>(m_animation_name);

		auto *cmd_buffer = rhi_context->CreateCommand(RHIQueueFamily::Graphics);
		cmd_buffer->Begin();

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

		{
			VertexInputState vertex_input_state = {};
			vertex_input_state.input_bindings   = {
                VertexInputState::InputBinding{0, sizeof(Resource<ResourceType::SkinnedMesh>::SkinnedVertex), RHIVertexInputRate::Vertex}};
			vertex_input_state.input_attributes = {
			    VertexInputState::InputAttribute{RHIVertexSemantics::Position, 0, 0, RHIFormat::R32G32B32_FLOAT, offsetof(Resource<ResourceType::SkinnedMesh>::SkinnedVertex, position)},
			    VertexInputState::InputAttribute{RHIVertexSemantics::Blend_Indices, 1, 0, RHIFormat::R32G32B32A32_SINT, offsetof(Resource<ResourceType::SkinnedMesh>::SkinnedVertex, bones[0])},
			    VertexInputState::InputAttribute{RHIVertexSemantics::Blend_Indices, 2, 0, RHIFormat::R32G32B32A32_SINT, offsetof(Resource<ResourceType::SkinnedMesh>::SkinnedVertex, bones[4])},
			    VertexInputState::InputAttribute{RHIVertexSemantics::Blend_Weights, 3, 0, RHIFormat::R32G32B32A32_FLOAT, offsetof(Resource<ResourceType::SkinnedMesh>::SkinnedVertex, weights[0])},
			    VertexInputState::InputAttribute{RHIVertexSemantics::Blend_Weights, 4, 0, RHIFormat::R32G32B32A32_FLOAT, offsetof(Resource<ResourceType::SkinnedMesh>::SkinnedVertex, weights[4])},
			};

			InputAssemblyState input_assembly_state = {};
			input_assembly_state.topology           = RHIPrimitiveTopology::Triangle;

			RasterizationState rasterization_state = {};
			rasterization_state.polygon_mode       = RHIPolygonMode::Solid;
			rasterization_state.line_width         = 1.f;

			m_pipeline_state->SetVertexInputState(vertex_input_state);
			m_pipeline_state->SetInputAssemblyState(input_assembly_state);
			m_pipeline_state->SetRasterizationState(rasterization_state);

			auto *vertex_shader   = p_editor->GetRenderer()->RequireShader("./Source/Shaders/Editor/AnimationEditor.hlsl", "VSmain", RHIShaderStage::Vertex);
			auto *fragment_shader = p_editor->GetRenderer()->RequireShader("./Source/Shaders/Editor/AnimationEditor.hlsl", "PSmain", RHIShaderStage::Fragment);

			m_pipeline_state->ClearShader();
			m_pipeline_state->SetShader(RHIShaderStage::Vertex, vertex_shader);
			m_pipeline_state->SetShader(RHIShaderStage::Fragment, fragment_shader);

			ShaderMeta meta = p_editor->GetRenderer()->RequireShaderMeta(vertex_shader);
			meta += p_editor->GetRenderer()->RequireShaderMeta(fragment_shader);

			m_render_target->Clear();
			m_render_target->Set(0, m_render_texture.get(), TextureRange{RHITextureDimension::Texture2D, 0, 1, 0, 1}, ColorAttachment{RHILoadAction::Clear, RHIStoreAction::Store});
			m_render_target->Set(m_depth_texture.get(), TextureRange{RHITextureDimension::Texture2D, 0, 1, 0, 1}, DepthStencilAttachment{});

			auto *descriptor = rhi_context->CreateDescriptor(meta);
			descriptor->BindBuffer("UniformBuffer", m_uniform_buffer.get())
			    .BindBuffer("BoneMatrices", m_skinned_buffer.get());

			cmd_buffer->BeginRenderPass(m_render_target.get());
			cmd_buffer->BindDescriptor(descriptor);
			cmd_buffer->BindPipelineState(m_pipeline_state.get());
			cmd_buffer->SetViewport((float) m_render_target->GetWidth(), (float) m_render_target->GetHeight());
			cmd_buffer->SetScissor(m_render_target->GetWidth(), m_render_target->GetHeight());

			for (auto &skinned_mesh_name : m_skinned_meshes)
			{
				auto *skinned_mesh = p_editor->GetRenderer()->GetResourceManager()->Get<ResourceType::SkinnedMesh>(skinned_mesh_name);
				if (skinned_mesh)
				{
					cmd_buffer->BindVertexBuffer(0, skinned_mesh->GetVertexBuffer());
					cmd_buffer->BindIndexBuffer(skinned_mesh->GetIndexBuffer());
					cmd_buffer->DrawIndexed(static_cast<uint32_t>(skinned_mesh->GetIndexCount()));
				}
			}

			cmd_buffer->EndRenderPass();
		}

		cmd_buffer->ResourceStateTransition({TextureStateTransition{m_render_texture.get(), RHIResourceState::RenderTarget, RHIResourceState::ShaderResource, TextureRange{RHITextureDimension::Texture2D, 0, 1, 0, 1}}}, {});
		cmd_buffer->End();
		rhi_context->Submit({cmd_buffer});
	}

  private:
	std::string m_animation_name = "";

	std::vector<std::string> m_skinned_meshes;

	HierarchyNode m_root;

	std::unique_ptr<RHIPipelineState> m_pipeline_state = nullptr;
	std::unique_ptr<RHIRenderTarget>  m_render_target  = nullptr;

	std::unique_ptr<RHITexture> m_render_texture = nullptr;
	std::unique_ptr<RHITexture> m_depth_texture  = nullptr;

	bool m_wireframe = false;

	bool m_start = false;

	std::unique_ptr<RHIBuffer> m_uniform_buffer = nullptr;
	std::unique_ptr<RHIBuffer> m_skinned_buffer = nullptr;
	std::unique_ptr<RHIBuffer> m_vertex_buffer  = nullptr;

	float m_scale_factor = 1.f;

	float m_time = 0.f;

	struct
	{
		glm::vec3 center = glm::vec3(0.f);
		float     radius = 0.f;
		float     theta  = 0.f;
		float     phi    = 90.f;
	} m_view;
};

extern "C"
{
	EXPORT_API AnimationEditor *Create(Editor *editor, ImGuiContext *context)
	{
		ImGui::SetCurrentContext(context);
		return new AnimationEditor(editor);
	}
}