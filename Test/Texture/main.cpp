#include "Engine/Context.hpp"
#include "Engine/Engine.hpp"

#include "Timing/Timer.hpp"

#include "Device/Surface.hpp"
#include "Device/Swapchain.hpp"
#include "Device/Window.hpp"
#include <Device/Input.hpp>

#include "Renderer/RenderGraph/RenderPass.hpp"
#include "Renderer/RenderPass/DebugPass.hpp"
#include "Renderer/RenderPass/ImGuiPass.hpp"
#include "Renderer/Renderer.hpp"

#include "Graphics/GraphicsContext.hpp"
#include "Graphics/Pipeline/PipelineState.hpp"

#include "Loader/ImageLoader/ImageLoader.hpp"

#include <Editor/Editor.hpp>

#include <glm/glm.hpp>

using namespace Ilum;

class TexturePass : public TRenderPass<TexturePass>
{
  public:
	TexturePass()
	{
		vertex_buffer = Buffer(vertices.size() * sizeof(Vertex), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
		index_buffer  = Buffer(indices.size() * sizeof(uint16_t), VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);

		{
			Buffer staging_buffer(sizeof(Vertex) * vertices.size(), VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
			auto * data = staging_buffer.map();
			std::memcpy(data, vertices.data(), sizeof(Vertex) * vertices.size());
			staging_buffer.unmap();
			CommandBuffer command_buffer;
			command_buffer.begin();
			command_buffer.copyBuffer(BufferInfo{staging_buffer}, BufferInfo{vertex_buffer}, sizeof(Vertex) * vertices.size());
			command_buffer.end();
			command_buffer.submitIdle();
		}

		{
			Buffer staging_buffer(sizeof(uint16_t) * indices.size(), VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
			auto * data = staging_buffer.map();
			std::memcpy(data, indices.data(), sizeof(uint16_t) * indices.size());
			staging_buffer.unmap();
			CommandBuffer command_buffer;
			command_buffer.begin();
			command_buffer.copyBuffer(BufferInfo{staging_buffer}, BufferInfo{index_buffer}, sizeof(uint16_t) * indices.size());
			command_buffer.end();
			command_buffer.submitIdle();
		}
	}

	virtual void setupPipeline(PipelineState &state)
	{
		state.shader.load(std::string(PROJECT_SOURCE_DIR) + "Test/Texture/texture.glsl.vert", VK_SHADER_STAGE_VERTEX_BIT, Shader::Type::GLSL);
		state.shader.load(std::string(PROJECT_SOURCE_DIR) + "Test/Texture/texture.glsl.frag", VK_SHADER_STAGE_FRAGMENT_BIT, Shader::Type::GLSL);

		state.dynamic_state.dynamic_states = {
		    VK_DYNAMIC_STATE_VIEWPORT,
		    VK_DYNAMIC_STATE_SCISSOR};

		state.vertex_input_state.attribute_descriptions = {
		    VkVertexInputAttributeDescription{0, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(Vertex, pos)},
		    VkVertexInputAttributeDescription{1, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(Vertex, uv)},
		    VkVertexInputAttributeDescription{2, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, color)}};

		state.vertex_input_state.binding_descriptions = {
		    VkVertexInputBindingDescription{0, sizeof(Vertex), VK_VERTEX_INPUT_RATE_VERTEX}};

		state.descriptor_bindings.bind(0, 0, "tex", Renderer::instance()->getSampler(Renderer::SamplerType::Trilinear_Clamp), ImageViewType::Native, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);

		state.declareAttachment("output", GraphicsContext::instance()->getSurface().getFormat().format);
		state.declareAttachment("depth_stencil", VK_FORMAT_D32_SFLOAT_S8_UINT);

		state.addOutputAttachment("output", AttachmentState::Clear_Color);
		state.addOutputAttachment("depth_stencil", AttachmentState::Clear_Depth_Stencil);
	}

	virtual void resolveResources(ResolveState &resolve)
	{
		resolve.resolve("tex", Renderer::instance()->getResourceCache().loadImage("../Test/Texture/texture.jpg"));
	};

	virtual void render(RenderPassState &state)
	{
		auto &cmd_buffer = state.command_buffer;

		auto &extent = GraphicsContext::instance()->getSwapchain().getExtent();

		VkViewport viewport = {0, 0, static_cast<float>(extent.width), static_cast<float>(extent.height), 0, 1};
		VkRect2D   scissor  = {0, 0, extent.width, extent.height};

		vkCmdSetViewport(cmd_buffer, 0, 1, &viewport);
		vkCmdSetScissor(cmd_buffer, 0, 1, &scissor);

		VkDeviceSize offsets[1] = {0};
		vkCmdBindVertexBuffers(cmd_buffer, 0, 1, &vertex_buffer.getBuffer(), offsets);
		vkCmdBindIndexBuffer(cmd_buffer, index_buffer.getBuffer(), 0, VK_INDEX_TYPE_UINT16);
		vkCmdDrawIndexed(cmd_buffer, static_cast<uint32_t>(indices.size()), 1, 0, 0, 0);
	};

  private:
	struct Vertex
	{
		glm::vec2 pos;
		glm::vec2 uv;
		glm::vec3 color;
	};

	const std::vector<Vertex> vertices = {
	    {{-0.5f, 0.5f}, {0.f, 1.f}, {1.0f, 0.0f, 0.0f}},
	    {{0.5f, 0.5f}, {1.f, 1.f}, {0.0f, 1.0f, 0.0f}},
	    {{0.5f, -0.5f}, {1.f, 0.f}, {0.0f, 0.0f, 1.0f}},
	    {{-0.5f, -0.5f}, {0.f, 0.f}, {0.0f, 0.0f, 1.0f}}};

	const std::vector<uint16_t> indices = {
	    0, 1, 3, 3, 1, 2};

	Buffer vertex_buffer;
	Buffer index_buffer;
};

int main()
{
	Engine engine;

	auto title = Window::instance()->getTitle();

	Renderer::instance()->setDebug(false);
	Renderer::instance()->setImGui(false);

	Renderer::instance()->buildRenderGraph = [](RenderGraphBuilder &builder) {
		builder.addRenderPass("TexturePass", std::make_unique<TexturePass>()).setView("output").setOutput("output");
	};

	Renderer::instance()->rebuild();

	while (!Window::instance()->shouldClose())
	{
		engine.onTick();

		Ilum::Window::instance()->setTitle(title + " FPS: " + std::to_string(Timer::instance()->getFPS()));
	}

	return 0;
}