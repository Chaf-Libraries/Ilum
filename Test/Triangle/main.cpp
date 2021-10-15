#include "Engine/Context.hpp"
#include "Engine/Engine.hpp"

#include "Timing/Timer.hpp"

#include "Device/Surface.hpp"
#include "Device/Swapchain.hpp"
#include "Device/Window.hpp"
#include <Device/Input.hpp>

#include "Renderer/RenderGraph/RenderPass.hpp"
#include "Renderer/RenderPass/ImGuiPass.hpp"
#include "Renderer/Renderer.hpp"

#include "Graphics/GraphicsContext.hpp"
#include "Graphics/Pipeline/PipelineState.hpp"

#include "Math/Vector2.h"
#include "Math/Vector3.h"

#include "Loader/ImageLoader/ImageLoader.hpp"

#include <Editor/Editor.hpp>

using namespace Ilum;

class TrianglePass : public TRenderPass<TrianglePass>
{
  public:
	TrianglePass()
	{
		vertex_buffer = createScope<Buffer>(vertices.size() * sizeof(Vertex), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
		index_buffer  = createScope<Buffer>(indices.size() * sizeof(uint16_t), VK_BUFFER_USAGE_INDEX_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);

		auto *vertex_data = vertex_buffer->map();
		std::memcpy(vertex_data, vertices.data(), vertex_buffer->getSize());
		vertex_buffer->unmap();

		auto *index_data = index_buffer->map();
		std::memcpy(index_data, indices.data(), index_buffer->getSize());
		index_buffer->unmap();
	}

	virtual void setupPipeline(PipelineState &state)
	{
		state.shader.createShaderModule(std::string(PROJECT_SOURCE_DIR) + "Test/Triangle/triangle.glsl.vert");
		state.shader.createShaderModule(std::string(PROJECT_SOURCE_DIR) + "Test/Triangle/triangle.glsl.frag");

		state.dynamic_state.dynamic_states = {
		    VK_DYNAMIC_STATE_VIEWPORT,
		    VK_DYNAMIC_STATE_SCISSOR};

		state.vertex_input_state.attribute_descriptions = {
		    VkVertexInputAttributeDescription{0, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(Vertex, pos)},
		    VkVertexInputAttributeDescription{1, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(Vertex, uv)},
		    VkVertexInputAttributeDescription{2, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, color)}};

		state.vertex_input_state.binding_descriptions = {
		    VkVertexInputBindingDescription{0, sizeof(Vertex), VK_VERTEX_INPUT_RATE_VERTEX}};

		state.declareAttachment("output", GraphicsContext::instance()->getSurface().getFormat().format);

		state.addOutputAttachment("output", AttachmentState::Clear_Color);
	}

	virtual void resolveResources(ResolveState &resolve){

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
		vkCmdBindVertexBuffers(cmd_buffer, 0, 1, &vertex_buffer->getBuffer(), offsets);
		vkCmdBindIndexBuffer(cmd_buffer, index_buffer->getBuffer(), 0, VK_INDEX_TYPE_UINT16);
		vkCmdDrawIndexed(cmd_buffer, static_cast<uint32_t>(indices.size()), 1, 0, 0, 0);
	};

  private:
	struct Vertex
	{
		Vector2 pos;
		Vector2 uv;
		Vector3 color;
	};

	const std::vector<Vertex> vertices = {
	    {{-0.5f, 0.5f}, {-0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}},
	    {{0.5f, 0.5f}, {0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}},
	    {{0.f, -0.5f}, {0.5f, 0.5f}, {0.0f, 0.0f, 1.0f}}};

	const std::vector<uint16_t> indices = {
	    0, 1, 2};

	scope<Buffer> vertex_buffer = nullptr;
	scope<Buffer> index_buffer  = nullptr;
};

int main()
{
	Engine engine;

	auto title = Window::instance()->getTitle();

	auto &builder = Renderer::instance()->getRenderGraphBuilder();
	builder.reset();

	Renderer::instance()->buildRenderGraph = [](RenderGraphBuilder &builder) {
		builder.addRenderPass("TrianglePass", std::make_unique<TrianglePass>()).setOutput("output");

		builder.addRenderPass("ImGuiPass", std::make_unique<ImGuiPass>("output", AttachmentState::Load_Color)).setOutput("output");
	};

	Renderer::instance()->rebuild();

	while (!Window::instance()->shouldClose())
	{
		engine.onTick();

		Ilum::Window::instance()->setTitle(title + " FPS: " + std::to_string(Timer::instance()->getFPS()));
	}

	return 0;
}