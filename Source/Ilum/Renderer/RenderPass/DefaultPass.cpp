#include "DefaultPass.hpp"

#include "Graphics/Model/Model.hpp"
#include "Graphics/Model/Vertex.hpp"
#include "Graphics/GraphicsContext.hpp"

#include "Device/Surface.hpp"

#include "Renderer/Renderer.hpp"

#include "Scene/Component/MeshRenderer.hpp"
#include "Scene/Component/Tag.hpp"
#include "Scene/Component/Transform.hpp"
#include "Scene/Entity.hpp"
#include "Scene/Scene.hpp"

#include "Material/BlinnPhong.h"

#include <glm/gtc/type_ptr.hpp>

namespace Ilum::pass
{
DefaultPass::DefaultPass(const std::string &output) :
    m_output(output)
{
}

void DefaultPass::setupPipeline(PipelineState &state)
{
	state.shader.load(std::string(PROJECT_SOURCE_DIR) + "Asset/Shader/GLSL/default.vert", VK_SHADER_STAGE_VERTEX_BIT, Shader::Type::GLSL);
	state.shader.load(std::string(PROJECT_SOURCE_DIR) + "Asset/Shader/GLSL/default.frag", VK_SHADER_STAGE_FRAGMENT_BIT, Shader::Type::GLSL);

	state.dynamic_state.dynamic_states = {
	    VK_DYNAMIC_STATE_VIEWPORT,
	    VK_DYNAMIC_STATE_SCISSOR};

	state.vertex_input_state.attribute_descriptions = {
	    VkVertexInputAttributeDescription{0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, position)},
	    VkVertexInputAttributeDescription{1, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(Vertex, texcoord)},
	    VkVertexInputAttributeDescription{2, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, normal)},
	    VkVertexInputAttributeDescription{3, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, tangent)},
	    VkVertexInputAttributeDescription{4, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, bitangent)}};

	state.vertex_input_state.binding_descriptions = {
	    VkVertexInputBindingDescription{0, sizeof(Vertex), VK_VERTEX_INPUT_RATE_VERTEX}};

	state.descriptor_bindings.bind(0, 0, "DefaultPass - mainCamera", VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);

	state.declareAttachment(m_output, VK_FORMAT_R8G8B8A8_UNORM, Renderer::instance()->getRenderTargetExtent().width, Renderer::instance()->getRenderTargetExtent().height);
	state.declareAttachment("depth_stencil", VK_FORMAT_D32_SFLOAT_S8_UINT, Renderer::instance()->getRenderTargetExtent().width, Renderer::instance()->getRenderTargetExtent().height);

	state.addOutputAttachment(m_output, AttachmentState::Clear_Color);
	state.addOutputAttachment("depth_stencil", VkClearDepthStencilValue{1.f, 0u});
}

void DefaultPass::resolveResources(ResolveState &resolve)
{
	resolve.resolve("DefaultPass - mainCamera", Renderer::instance()->getBuffer(Renderer::BufferType::MainCamera));
}

void DefaultPass::render(RenderPassState &state)
{
	auto &cmd_buffer = state.command_buffer;

	auto &extent = Renderer::instance()->getRenderTargetExtent();

	VkViewport viewport = {0, static_cast<float>(extent.height), static_cast<float>(extent.width), -static_cast<float>(extent.height), 0, 1};
	VkRect2D   scissor  = {0, 0, extent.width, extent.height};

	vkCmdSetViewport(cmd_buffer, 0, 1, &viewport);
	vkCmdSetScissor(cmd_buffer, 0, 1, &scissor);

	const auto view = Scene::instance()->getRegistry().view<cmpt::MeshRenderer, cmpt::Transform, cmpt::Tag>();

	view.each([&](const cmpt::MeshRenderer &mesh_renderer, const cmpt::Transform &transform, const cmpt::Tag &tag) {
		if (Renderer::instance()->getResourceCache().hasModel(mesh_renderer.model) && tag.active)
		{
			auto &model = Renderer::instance()->getResourceCache().loadModel(mesh_renderer.model);

			VkDeviceSize offsets[1] = {0};
			vkCmdBindVertexBuffers(cmd_buffer, 0, 1, &model.get().getVertexBuffer().get().getBuffer(), offsets);
			vkCmdBindIndexBuffer(cmd_buffer, model.get().getIndexBuffer().get().getBuffer(), 0, VK_INDEX_TYPE_UINT32);

			// Model transform push constants
			vkCmdPushConstants(cmd_buffer, state.pass.pipeline_layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(glm::mat4), glm::value_ptr(transform.world_transform));

			for (uint32_t i = 0; i < model.get().getSubMeshes().size(); i++)
			{
				if (!mesh_renderer.materials[i])
				{
					const auto &submesh = model.get().getSubMeshes()[i];
					vkCmdDrawIndexed(cmd_buffer, submesh.getIndexCount(), 1, submesh.getIndexOffset(), 0, 0);
				}
			}
		}
	});
}
}        // namespace Ilum::pass