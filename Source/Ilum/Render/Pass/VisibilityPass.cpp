#include "VisibilityPass.hpp"

#include <RHI/DescriptorState.hpp>
#include <RHI/FrameBuffer.hpp>

#include <Render/RGBuilder.hpp>
#include <Render/Renderer.hpp>

#include <Scene/Component/MeshRenderer.hpp>
#include <Scene/Scene.hpp>

namespace Ilum
{
VisibilityPass::VisibilityPass() :
    RenderPass("VisibilityPass")
{
}

void VisibilityPass::Create(RGBuilder &builder)
{
	std::unique_ptr<RenderPass> pass = std::make_unique<VisibilityPass>();

	auto debug_buffer = builder.CreateBuffer(
	    "Debug",
	    BufferDesc{
	        sizeof(uint32_t),
	        10000,
	        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
	        VMA_MEMORY_USAGE_GPU_TO_CPU},
	    BufferState{VK_BUFFER_USAGE_STORAGE_BUFFER_BIT});

	auto visbility_buffer = builder.CreateTexture(
	    "Visibility Buffer",
	    TextureDesc{
	        builder.GetRenderer().GetExtent().width,  /*width*/
	        builder.GetRenderer().GetExtent().height, /*height*/
	        1,                                        /*depth*/
	        1,                                        /*mips*/
	        1,                                        /*layers*/
	        VK_SAMPLE_COUNT_1_BIT,
	        VK_FORMAT_R8G8B8A8_UNORM,
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

	pass->AddResource(debug_buffer);
	pass->AddResource(visbility_buffer);
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

	ShaderDesc task_shader  = {};
	task_shader.filename    = "./Source/Shaders/Shading/Visibility.hlsl";
	task_shader.entry_point = "ASmain";
	task_shader.stage       = VK_SHADER_STAGE_TASK_BIT_NV;
	task_shader.type        = ShaderType::HLSL;

	ShaderDesc mesh_shader  = {};
	mesh_shader.filename    = "./Source/Shaders/Shading/Visibility.hlsl";
	mesh_shader.entry_point = "MSmain";
	mesh_shader.stage       = VK_SHADER_STAGE_MESH_BIT_NV;
	mesh_shader.type        = ShaderType::HLSL;

	ShaderDesc frag_shader  = {};
	frag_shader.filename    = "./Source/Shaders/Shading/Visibility.hlsl";
	frag_shader.entry_point = "PSmain";
	frag_shader.stage       = VK_SHADER_STAGE_FRAGMENT_BIT;
	frag_shader.type        = ShaderType::HLSL;

	DynamicState dynamic_state = {};
	dynamic_state.dynamic_states.push_back(VK_DYNAMIC_STATE_SCISSOR);
	dynamic_state.dynamic_states.push_back(VK_DYNAMIC_STATE_VIEWPORT);

	ColorBlendState color_blend_state = {};
	color_blend_state.attachment_states.push_back(ColorBlendAttachmentState{});

	/*VertexInputState vertex_input_state       = {};
	vertex_input_state.attribute_descriptions = {
	    VkVertexInputAttributeDescription{0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(ShaderInterop::Vertex, position)},
	    VkVertexInputAttributeDescription{1, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(ShaderInterop::Vertex, texcoord)},
	    VkVertexInputAttributeDescription{2, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(ShaderInterop::Vertex, normal)},
	    VkVertexInputAttributeDescription{3, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(ShaderInterop::Vertex, tangent)}};
	vertex_input_state.binding_descriptions = {
	    VkVertexInputBindingDescription{0, sizeof(ShaderInterop::Vertex), VK_VERTEX_INPUT_RATE_VERTEX}};*/

	PipelineState pso;
	pso.SetDynamicState(dynamic_state);
	pso.SetColorBlendState(color_blend_state);
	// pso.SetVertexInputState(vertex_input_state);

	pso.LoadShader(task_shader);
	pso.LoadShader(mesh_shader);
	pso.LoadShader(frag_shader);

	pass->BindCallback([=](CommandBuffer &cmd_buffer, const RGResources &resource, Renderer &renderer) {
		auto *scene = renderer.GetScene();
		if (!scene)
		{
			return;
		}

		std::vector<uint32_t> debug_data(10000);
		std::memcpy(debug_data.data(), resource.GetBuffer(debug_buffer)->Map(), 10000 * sizeof(uint32_t));
		resource.GetBuffer(debug_buffer)->Unmap();

		FrameBuffer framebuffer;
		framebuffer.Bind(resource.GetTexture(visbility_buffer), vbuffer_view_desc, ColorAttachmentInfo{});
		framebuffer.Bind(resource.GetTexture(depth_stencil), depth_stencil_view_desc, DepthStencilAttachmentInfo{});
		cmd_buffer.BeginRenderPass(framebuffer);
		cmd_buffer.Bind(pso);

		auto mesh_view = scene->GetRegistry().view<cmpt::MeshRenderer>();

		std::vector<Buffer *> instances;
		std::vector<Buffer *> vertices;
		std::vector<Buffer *> meshlets;
		std::vector<Buffer *> meshlet_vertices;
		std::vector<Buffer *> meshlet_triangles;
		instances.reserve(mesh_view.size());
		vertices.reserve(mesh_view.size());
		meshlets.reserve(mesh_view.size());
		meshlet_vertices.reserve(mesh_view.size());
		meshlet_triangles.reserve(mesh_view.size());

		uint32_t meshlet_count = 0;

		if (!mesh_view.empty())
		{
			mesh_view.each([&](cmpt::MeshRenderer &mesh_renderer) {
				if (mesh_renderer.mesh)
				{
					instances.push_back(mesh_renderer.buffer.get());
					vertices.push_back(&mesh_renderer.mesh->GetVertexBuffer());
					meshlets.push_back(&mesh_renderer.mesh->GetMeshletBuffer());
					meshlet_vertices.push_back(&mesh_renderer.mesh->GetMeshletVertexBuffer());
					meshlet_triangles.push_back(&mesh_renderer.mesh->GetMeshletTriangleBuffer());
					meshlet_count += mesh_renderer.mesh->GetMeshletsCount();
				}
			});
			cmd_buffer.Bind(
			    cmd_buffer.GetDescriptorState()
			        .Bind(0, 0, &renderer.GetScene()->GetMainCameraBuffer())
			        .Bind(0, 1, instances)
			        .Bind(0, 2, meshlets)
			        .Bind(0, 3, vertices)
			        .Bind(0, 4, meshlet_vertices)
			        .Bind(0, 5, meshlet_triangles)
			        .Bind(0, 6, resource.GetBuffer(debug_buffer)));

			cmd_buffer.SetViewport(static_cast<float>(renderer.GetExtent().width), -static_cast<float>(renderer.GetExtent().height), 0, static_cast<float>(renderer.GetExtent().height));
			cmd_buffer.SetScissor(renderer.GetExtent().width, renderer.GetExtent().height);

			uint32_t instance_id = 0;
			mesh_view.each([&](cmpt::MeshRenderer &mesh_renderer) {
				if (mesh_renderer.mesh)
				{

					uint32_t meshlet_count = mesh_renderer.mesh->GetMeshletsCount();
					cmd_buffer.PushConstants(VK_SHADER_STAGE_MESH_BIT_NV | VK_SHADER_STAGE_TASK_BIT_NV, &instance_id, sizeof(instance_id), 0);
					cmd_buffer.PushConstants(VK_SHADER_STAGE_MESH_BIT_NV | VK_SHADER_STAGE_TASK_BIT_NV, &meshlet_count, sizeof(meshlet_count), sizeof(instance_id));
					std::chrono::time_point<std::chrono::high_resolution_clock> start_time = std::chrono::high_resolution_clock::now();
					vkCmdDrawMeshTasksNV(cmd_buffer, (meshlet_count + 32 - 1) / 32, 0);
					LOG_INFO("{} ms", std::chrono::duration<float, std::milli>(std::chrono::high_resolution_clock::now() - start_time).count());

					instance_id++;
				}
			});
		}
		cmd_buffer.EndRenderPass();
	});

	pass->BindImGui([=](ImGuiContext &, const RGResources &) {

	});

	builder.AddPass(std::move(pass));
}

}        // namespace Ilum