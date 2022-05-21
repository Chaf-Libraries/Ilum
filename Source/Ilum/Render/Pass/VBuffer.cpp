#include "VBuffer.hpp"

#include <RHI/DescriptorState.hpp>
#include <RHI/FrameBuffer.hpp>

#include <Render/RGBuilder.hpp>
#include <Render/Renderer.hpp>

#include <Scene/Component/Camera.hpp>
#include <Scene/Component/MeshRenderer.hpp>
#include <Scene/Entity.hpp>
#include <Scene/Scene.hpp>

#include <Asset/AssetManager.hpp>

namespace Ilum
{
VBuffer::VBuffer() :
    RenderPass("VBuffer")
{
}

void VBuffer::Create(RGBuilder &builder)
{
	std::unique_ptr<RenderPass> pass = std::make_unique<VBuffer>();

	// Render Target
	auto visibility_buffer = builder.CreateTexture(
	    "Visibility Buffer",
	    TextureDesc{
	        builder.GetRenderer().GetExtent().width,  /*width*/
	        builder.GetRenderer().GetExtent().height, /*height*/
	        1,                                        /*depth*/
	        1,                                        /*mips*/
	        1,                                        /*layers*/
	        VK_SAMPLE_COUNT_1_BIT,
	        VK_FORMAT_R32_UINT,
	        VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT},
	    TextureState{VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT});

	auto depth_stencil = builder.CreateTexture(
	    "Depth Stencil",
	    TextureDesc{
	        builder.GetRenderer().GetExtent().width,  /*width*/
	        builder.GetRenderer().GetExtent().height, /*height*/
	        1,                                        /*depth*/
	        1,                                        /*mips*/
	        1,                                        /*layers*/
	        VK_SAMPLE_COUNT_1_BIT,
	        VK_FORMAT_D32_SFLOAT_S8_UINT,
	        VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT},
	    TextureState{VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT});

	pass->AddResource(visibility_buffer);
	pass->AddResource(depth_stencil);

	TextureViewDesc vbuffer_view_desc  = {};
	vbuffer_view_desc.aspect           = VK_IMAGE_ASPECT_COLOR_BIT;
	vbuffer_view_desc.view_type        = VK_IMAGE_VIEW_TYPE_2D;
	vbuffer_view_desc.base_array_layer = 0;
	vbuffer_view_desc.base_mip_level   = 0;
	vbuffer_view_desc.layer_count      = 1;
	vbuffer_view_desc.level_count      = 1;

	TextureViewDesc depth_stencil_view_desc  = {};
	depth_stencil_view_desc.aspect           = VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT;
	depth_stencil_view_desc.view_type        = VK_IMAGE_VIEW_TYPE_2D;
	depth_stencil_view_desc.base_array_layer = 0;
	depth_stencil_view_desc.base_mip_level   = 0;
	depth_stencil_view_desc.layer_count      = 1;
	depth_stencil_view_desc.level_count      = 1;

	// Set Shader
	ShaderDesc task_shader  = {};
	task_shader.filename    = "./Source/Shaders/VBuffer.hlsl";
	task_shader.entry_point = "ASmain";
	task_shader.stage       = VK_SHADER_STAGE_TASK_BIT_NV;
	task_shader.type        = ShaderType::HLSL;

	ShaderDesc mesh_shader  = {};
	mesh_shader.filename    = "./Source/Shaders/VBuffer.hlsl";
	mesh_shader.entry_point = "MSmain";
	mesh_shader.stage       = VK_SHADER_STAGE_MESH_BIT_NV;
	mesh_shader.type        = ShaderType::HLSL;

	ShaderDesc opaque_frag_shader  = {};
	opaque_frag_shader.filename    = "./Source/Shaders/VBuffer.hlsl";
	opaque_frag_shader.entry_point = "PSmain";
	opaque_frag_shader.stage       = VK_SHADER_STAGE_FRAGMENT_BIT;
	opaque_frag_shader.type        = ShaderType::HLSL;

	ShaderDesc alpha_frag_shader  = {};
	alpha_frag_shader.filename    = "./Source/Shaders/VBuffer.hlsl";
	alpha_frag_shader.entry_point = "PSmain";
	alpha_frag_shader.stage       = VK_SHADER_STAGE_FRAGMENT_BIT;
	alpha_frag_shader.type        = ShaderType::HLSL;
	alpha_frag_shader.macros.push_back("ALPHA_TEST");

	DynamicState dynamic_state = {};
	dynamic_state.dynamic_states.push_back(VK_DYNAMIC_STATE_SCISSOR);
	dynamic_state.dynamic_states.push_back(VK_DYNAMIC_STATE_VIEWPORT);

	ColorBlendState blend_state = {};
	blend_state.attachment_states.resize(1);
	blend_state.attachment_states.back().blend_enable = VK_FALSE;

	RasterizationState rasterization_state = {};
	rasterization_state.cull_mode          = VK_CULL_MODE_NONE;

	/*VertexInputState vertex_input_state       = {};
	vertex_input_state.attribute_descriptions = {
	    VkVertexInputAttributeDescription{0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(ShaderInterop::Vertex, position)},
	    VkVertexInputAttributeDescription{1, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(ShaderInterop::Vertex, texcoord)},
	    VkVertexInputAttributeDescription{2, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(ShaderInterop::Vertex, normal)},
	    VkVertexInputAttributeDescription{3, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(ShaderInterop::Vertex, tangent)}};
	vertex_input_state.binding_descriptions = {
	    VkVertexInputBindingDescription{0, sizeof(ShaderInterop::Vertex), VK_VERTEX_INPUT_RATE_VERTEX}};*/

	PipelineState opaque_pso;
	opaque_pso
	    .SetName("VBuffer - Opaque")
	    .SetDynamicState(dynamic_state)
	    .SetColorBlendState(blend_state)
	    .SetRasterizationState(rasterization_state)
	    .LoadShader(task_shader)
	    .LoadShader(mesh_shader)
	    .LoadShader(opaque_frag_shader);

	PipelineState alpha_pso;
	alpha_pso
	    .SetName("VBuffer - Alpha")
	    .SetDynamicState(dynamic_state)
	    .SetColorBlendState(blend_state)
	    .SetRasterizationState(rasterization_state)
	    .LoadShader(task_shader)
	    .LoadShader(mesh_shader)
	    .LoadShader(alpha_frag_shader);

	pass->BindCallback([=](CommandBuffer &cmd_buffer, const RGResources &resource, Renderer &renderer) {
		auto *scene = renderer.GetScene();
		if (!scene || scene->GetInstanceBuffer().empty())
		{
			return;
		}

		Entity camera_entity = Entity(*scene, scene->GetMainCamera());
		if (!camera_entity.IsValid())
		{
			return;
		}

		auto *camera_buffer = camera_entity.GetComponent<cmpt::Camera>().GetBuffer();
		if (!camera_buffer)
		{
			return;
		}

		FrameBuffer         framebuffer;
		ColorAttachmentInfo attachment_info   = {};
		attachment_info.clear_value.uint32[0] = 0xffffffff;
		framebuffer.Bind(resource.GetTexture(visibility_buffer), vbuffer_view_desc, attachment_info);
		framebuffer.Bind(resource.GetTexture(depth_stencil), depth_stencil_view_desc, DepthStencilAttachmentInfo{});
		cmd_buffer.BeginRenderPass(framebuffer);

		cmd_buffer.SetViewport(static_cast<float>(renderer.GetExtent().width), -static_cast<float>(renderer.GetExtent().height), 0, static_cast<float>(renderer.GetExtent().height));
		cmd_buffer.SetScissor(renderer.GetExtent().width, renderer.GetExtent().height);

		// Draw Opaque
		{
			auto batch = renderer.GetScene()->Batch(AlphaMode::Opaque);

			std::vector<Buffer *> instances;
			instances.reserve(batch.meshes.size());
			for (auto &mesh : batch.meshes)
			{
				instances.push_back(mesh->GetBuffer());
			}

			if (!instances.empty())
			{
				cmd_buffer.Bind(opaque_pso);
				cmd_buffer.Bind(
				    cmd_buffer.GetDescriptorState()
				        .Bind(0, 0, camera_buffer)
				        .Bind(1, 0, instances)
				        .Bind(1, 1, renderer.GetScene()->GetAssetManager().GetMeshletBuffer())
				        .Bind(1, 2, renderer.GetScene()->GetAssetManager().GetVertexBuffer())
				        .Bind(1, 3, renderer.GetScene()->GetAssetManager().GetMeshletVertexBuffer())
				        .Bind(1, 4, renderer.GetScene()->GetAssetManager().GetMeshletTriangleBuffer()));

				uint32_t instance_id = 0;
				for (auto &mesh : batch.meshes)
				{
					uint32_t meshlet_count = mesh->GetMesh()->GetMeshletsCount();
					cmd_buffer.PushConstants(VK_SHADER_STAGE_MESH_BIT_NV | VK_SHADER_STAGE_TASK_BIT_NV, &instance_id, sizeof(instance_id), 0);
					cmd_buffer.PushConstants(VK_SHADER_STAGE_MESH_BIT_NV | VK_SHADER_STAGE_TASK_BIT_NV, &meshlet_count, sizeof(meshlet_count), sizeof(instance_id));
					vkCmdDrawMeshTasksNV(cmd_buffer, (meshlet_count + 32 - 1) / 32, 0);
					instance_id++;
				}
			}
		}

		// Draw Alpha
		{
			auto batch = renderer.GetScene()->Batch(AlphaMode::Masked | AlphaMode::Blend);

			std::vector<Buffer *> instances;
			instances.reserve(batch.meshes.size());
			for (auto &mesh : batch.meshes)
			{
				instances.push_back(mesh->GetBuffer());
			}

			if (!instances.empty())
			{
				cmd_buffer.Bind(alpha_pso);
				cmd_buffer.Bind(
				    cmd_buffer.GetDescriptorState()
				        .Bind(0, 0, camera_buffer)
				        .Bind(1, 0, instances)
				        .Bind(1, 1, renderer.GetScene()->GetAssetManager().GetMeshletBuffer())
				        .Bind(1, 2, renderer.GetScene()->GetAssetManager().GetVertexBuffer())
				        .Bind(1, 3, renderer.GetScene()->GetAssetManager().GetMeshletVertexBuffer())
				        .Bind(1, 4, renderer.GetScene()->GetAssetManager().GetMeshletTriangleBuffer())
				        .Bind(2, 0, renderer.GetScene()->GetAssetManager().GetMaterialBuffer())
				        .Bind(2, 1, renderer.GetScene()->GetAssetManager().GetTextureViews())
				        .Bind(2, 2, renderer.GetSampler(SamplerType::TrilinearWarp)));

				for (uint32_t i = 0; i < batch.order.size(); i++)
				{
					uint32_t instance_id   = batch.order[i];
					uint32_t meshlet_count = batch.meshes[instance_id]->GetMesh()->GetMeshletsCount();
					cmd_buffer.PushConstants(VK_SHADER_STAGE_MESH_BIT_NV | VK_SHADER_STAGE_TASK_BIT_NV | VK_SHADER_STAGE_FRAGMENT_BIT, &instance_id, sizeof(instance_id), 0);
					cmd_buffer.PushConstants(VK_SHADER_STAGE_MESH_BIT_NV | VK_SHADER_STAGE_TASK_BIT_NV | VK_SHADER_STAGE_FRAGMENT_BIT, &meshlet_count, sizeof(meshlet_count), sizeof(instance_id));
					vkCmdDrawMeshTasksNV(cmd_buffer, (meshlet_count + 32 - 1) / 32, 0);
				}
			}
		}
		cmd_buffer.EndRenderPass();
	});

	pass->BindImGui([=](ImGuiContext &, const RGResources &) {

	});

	builder.AddPass(std::move(pass));
}

}        // namespace Ilum