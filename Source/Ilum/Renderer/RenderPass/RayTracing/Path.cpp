#include "Path.hpp"

#include "Renderer/RenderGraph/RenderGraph.hpp"
#include "Renderer/Renderer.hpp"

#include "Scene/Component/Camera.hpp"

#include "Graphics/Vulkan/VK_Debugger.h"

#include "ImGui/ImGuiContext.hpp"

#include <imgui.h>

namespace Ilum::pass
{
void Path::setupPipeline(PipelineState &state)
{
	state.shader.load(std::string(PROJECT_SOURCE_DIR) + "Source/Shaders/RayTracing/Path.hlsl", VK_SHADER_STAGE_RAYGEN_BIT_KHR, Shader::Type::HLSL, "main");

	state.shader.load(std::string(PROJECT_SOURCE_DIR) + "Source/Shaders/RayTracing/ClosestHit/Matte.hlsl", VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, Shader::Type::HLSL, "main");
	state.shader.load(std::string(PROJECT_SOURCE_DIR) + "Source/Shaders/RayTracing/ClosestHit/Plastic.hlsl", VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, Shader::Type::HLSL, "main");
	state.shader.load(std::string(PROJECT_SOURCE_DIR) + "Source/Shaders/RayTracing/ClosestHit/Metal.hlsl", VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, Shader::Type::HLSL, "main");
	state.shader.load(std::string(PROJECT_SOURCE_DIR) + "Source/Shaders/RayTracing/ClosestHit/Mirror.hlsl", VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, Shader::Type::HLSL, "main");
	state.shader.load(std::string(PROJECT_SOURCE_DIR) + "Source/Shaders/RayTracing/ClosestHit/Substrate.hlsl", VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, Shader::Type::HLSL, "main");
	state.shader.load(std::string(PROJECT_SOURCE_DIR) + "Source/Shaders/RayTracing/ClosestHit/Glass.hlsl", VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, Shader::Type::HLSL, "main");
	state.shader.load(std::string(PROJECT_SOURCE_DIR) + "Source/Shaders/RayTracing/ClosestHit/Disney.hlsl", VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, Shader::Type::HLSL, "main");

	state.shader.load(std::string(PROJECT_SOURCE_DIR) + "Source/Shaders/RayTracing/Miss/Matte.hlsl", VK_SHADER_STAGE_MISS_BIT_KHR, Shader::Type::HLSL, "main");
	state.shader.load(std::string(PROJECT_SOURCE_DIR) + "Source/Shaders/RayTracing/Miss/Plastic.hlsl", VK_SHADER_STAGE_MISS_BIT_KHR, Shader::Type::HLSL, "main");
	state.shader.load(std::string(PROJECT_SOURCE_DIR) + "Source/Shaders/RayTracing/Miss/Metal.hlsl", VK_SHADER_STAGE_MISS_BIT_KHR, Shader::Type::HLSL, "main");
	state.shader.load(std::string(PROJECT_SOURCE_DIR) + "Source/Shaders/RayTracing/Miss/Mirror.hlsl", VK_SHADER_STAGE_MISS_BIT_KHR, Shader::Type::HLSL, "main");
	state.shader.load(std::string(PROJECT_SOURCE_DIR) + "Source/Shaders/RayTracing/Miss/Substrate.hlsl", VK_SHADER_STAGE_MISS_BIT_KHR, Shader::Type::HLSL, "main");
	state.shader.load(std::string(PROJECT_SOURCE_DIR) + "Source/Shaders/RayTracing/Miss/Glass.hlsl", VK_SHADER_STAGE_MISS_BIT_KHR, Shader::Type::HLSL, "main");
	state.shader.load(std::string(PROJECT_SOURCE_DIR) + "Source/Shaders/RayTracing/Miss/Disney.hlsl", VK_SHADER_STAGE_MISS_BIT_KHR, Shader::Type::HLSL, "main");

	state.declareAttachment("Path", VK_FORMAT_R16G16B16A16_SFLOAT, Renderer::instance()->getRenderTargetExtent().width, Renderer::instance()->getRenderTargetExtent().height);
	state.addOutputAttachment("Path", AttachmentState::Clear_Color);

	state.declareAttachment("PrevPath", VK_FORMAT_R16G16B16A16_SFLOAT, Renderer::instance()->getRenderTargetExtent().width, Renderer::instance()->getRenderTargetExtent().height);
	state.addOutputAttachment("PrevPath", AttachmentState::Clear_Color);

	state.descriptor_bindings.bind(0, 0, "TLAS", VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR);
	state.descriptor_bindings.bind(0, 1, "Path", VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
	state.descriptor_bindings.bind(0, 2, "PrevPath", VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
	state.descriptor_bindings.bind(0, 3, "Camera", VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
	state.descriptor_bindings.bind(0, 4, "Vertices", VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
	state.descriptor_bindings.bind(0, 5, "Indices", VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
	state.descriptor_bindings.bind(0, 6, "PerInstanceBuffer", VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
	state.descriptor_bindings.bind(0, 7, "MaterialBuffer", VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
	state.descriptor_bindings.bind(0, 8, "TextureArray", Renderer::instance()->getSampler(Renderer::SamplerType::Trilinear_Wrap), ImageViewType::Native, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
	state.descriptor_bindings.bind(0, 9, "DirectionalLights", VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
	state.descriptor_bindings.bind(0, 10, "PointLights", VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
	state.descriptor_bindings.bind(0, 11, "SpotLights", VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
	state.descriptor_bindings.bind(0, 12, "SkyBox", Renderer::instance()->getSampler(Renderer::SamplerType::Trilinear_Clamp), ImageViewType::Cube, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
}

void Path::resolveResources(ResolveState &resolve)
{
	resolve.resolve("TLAS", Renderer::instance()->Render_Buffer.Top_Level_AS);
	resolve.resolve("Camera", Renderer::instance()->Render_Buffer.Camera_Buffer);
	resolve.resolve("Vertices", Renderer::instance()->Render_Buffer.Static_Vertex_Buffer);
	resolve.resolve("Indices", Renderer::instance()->Render_Buffer.Static_Index_Buffer);
	resolve.resolve("PerInstanceBuffer", Renderer::instance()->Render_Buffer.Instance_Buffer);
	resolve.resolve("MaterialBuffer", Renderer::instance()->Render_Buffer.Material_Buffer);
	resolve.resolve("TextureArray", Renderer::instance()->getResourceCache().getImageReferences());
	resolve.resolve("DirectionalLights", Renderer::instance()->Render_Buffer.Directional_Light_Buffer);
	resolve.resolve("PointLights", Renderer::instance()->Render_Buffer.Point_Light_Buffer);
	resolve.resolve("SpotLights", Renderer::instance()->Render_Buffer.Spot_Light_Buffer);
}

void Path::render(RenderPassState &state)
{
	if (!m_update)
	{
		return;
	}

	const auto &vertex_buffer = Renderer::instance()->Render_Buffer.Static_Vertex_Buffer;
	const auto &index_buffer  = Renderer::instance()->Render_Buffer.Static_Index_Buffer;

	if (Renderer::instance()->Render_Stats.static_mesh_count.meshlet_count > 0 && vertex_buffer.getBuffer() && index_buffer.getBuffer())
	{
		auto &cmd_buffer = state.command_buffer;

		vkCmdBindPipeline(cmd_buffer, state.pass.bind_point, state.pass.pipeline);

		for (auto &descriptor_set : state.pass.descriptor_sets)
		{
			vkCmdBindDescriptorSets(cmd_buffer, state.pass.bind_point, state.pass.pipeline_layout, descriptor_set.index(), 1, &descriptor_set.getDescriptorSet(), 0, nullptr);
		}

		m_push_block.directional_light_count = Renderer::instance()->Render_Stats.light_count.directional_light_count;
		m_push_block.spot_light_count        = Renderer::instance()->Render_Stats.light_count.spot_light_count;
		m_push_block.point_light_count       = Renderer::instance()->Render_Stats.light_count.point_light_count;

		vkCmdPushConstants(cmd_buffer, state.pass.pipeline_layout, VK_SHADER_STAGE_RAYGEN_BIT_KHR, 0, sizeof(m_push_block), &m_push_block);

		vkCmdTraceRaysKHR(
		    cmd_buffer,
		    state.pass.shader_binding_table.raygen->getHandle(),
		    state.pass.shader_binding_table.miss->getHandle(),
		    state.pass.shader_binding_table.hit->getHandle(),
		    state.pass.shader_binding_table.callable->getHandle(),
		    Renderer::instance()->getRenderTargetExtent().width,
		    Renderer::instance()->getRenderTargetExtent().height,
		    1);
	}
}

void Path::onImGui()
{
	auto &camera_entity = Renderer::instance()->Main_Camera;

	if (camera_entity && (camera_entity.hasComponent<cmpt::PerspectiveCamera>() || camera_entity.hasComponent<cmpt::OrthographicCamera>()))
	{
		cmpt::Camera *camera = camera_entity.hasComponent<cmpt::PerspectiveCamera>() ?
                                   static_cast<cmpt::Camera *>(&camera_entity.getComponent<cmpt::PerspectiveCamera>()) :
                                   static_cast<cmpt::Camera *>(&camera_entity.getComponent<cmpt::OrthographicCamera>());

		m_update = static_cast<int32_t>(camera->frame_count) - 1 < m_max_spp;

		camera->frame_count = glm::clamp(static_cast<int32_t>(camera->frame_count), 0, m_max_spp);

		ImGui::Text("SPP: %d", camera->frame_count);

		if (ImGui::Checkbox("Anti-Aliasing", reinterpret_cast<bool *>(&m_push_block.anti_alias)))
		{
			camera->frame_count = 0;
		}
		if (ImGui::SliderInt("Max Bounce", &m_push_block.max_bounce, 0, 100))
		{
			camera->frame_count = 0;
		}

		if (ImGui::DragInt("Max SPP", &m_max_spp, 0.1f, 0, std::numeric_limits<int32_t>::max()))
		{
			camera->frame_count = 0;
		}

		ImGui::ProgressBar(static_cast<float>(camera->frame_count) / static_cast<float>(m_max_spp),
		                   ImVec2(0.f, 0.f),
		                   (std::to_string(camera->frame_count) + "/" + std::to_string(m_max_spp)).c_str());

		if (ImGui::DragFloat("Clamp Threshold", &m_push_block.firefly_clamp_threshold, 0.1f, 0.0f, std::numeric_limits<float>::max()))
		{
			camera->frame_count = 0;
		}

		if (ImGui::DragFloat("Parameter", &m_push_block.parameter, 0.1f, 0.0f, std::numeric_limits<float>::max()))
		{
			camera->frame_count = 0;
		}
	}
}
}        // namespace Ilum::pass