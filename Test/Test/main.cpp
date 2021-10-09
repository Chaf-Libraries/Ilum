#include <Core/Device/Input.hpp>
#include <Core/Device/Window.hpp>
#include <Core/Engine/Context.hpp>
#include <Core/Engine/Engine.hpp>
#include <Core/Engine/File/FileSystem.hpp>
#include <Core/Engine/Pooling/MemoryPool.hpp>
#include <Core/Engine/Timing/Timer.hpp>

#include <Core/Device/Instance.hpp>
#include <Core/Device/LogicalDevice.hpp>
#include <Core/Device/PhysicalDevice.hpp>
#include <Core/Device/Surface.hpp>

#include <Core/Graphics/Buffer/Buffer.h>
#include <Core/Graphics/Descriptor/DescriptorCache.hpp>
#include <Core/Graphics/Descriptor/DescriptorLayout.hpp>
#include <Core/Graphics/GraphicsContext.hpp>
#include <Core/Graphics/Image/Image2D.hpp>
#include <Core/Graphics/Image/ImageDepth.hpp>
#include <Core/Graphics/Pipeline/PipelineGraphics.hpp>
#include <Core/Graphics/Pipeline/Shader.hpp>
#include <Core/Graphics/RenderPass/Framebuffer.hpp>
#include <Core/Graphics/RenderPass/RenderPass.hpp>
#include <Core/Graphics/RenderPass/RenderTarget.hpp>
#include <Core/Graphics/RenderPass/Swapchain.hpp>

#include <Core/Resource/Bitmap/Bitmap.hpp>

#include <Math/Vector2.h>
#include <Math/Vector3.h>
#include <Math/Vector4.h>

#include "Core/Geometry/Vertex.hpp"

struct Vertex
{
	Math::Vector4 pos;
	Math::Vector3 color;
	Math::Vector3 normal;
};

struct InstanceData
{
	uint32_t idx;
};

int main()
{
	auto engine = std::make_unique<Ilum::Engine>();

	//auto bitmap = Ilum::Bitmap::create("../Asset/Texture/613934.jpg");
	auto bitmap = Ilum::Bitmap::create("../Asset/Texture/hdr/circus_arena_4k.hdr");
	bitmap->write("test.hdr");

	auto image = Ilum::Image2D::create("../Asset/Texture/613934.jpg", VK_FILTER_LINEAR, VK_SAMPLER_ADDRESS_MODE_REPEAT, true);
	auto depth = std::make_unique<Ilum::ImageDepth>(100, 100, VK_SAMPLE_COUNT_1_BIT);

	const std::string title = Ilum::Window::instance()->getTitle();

	Vertex vert;

	Ilum::Buffer buffer(sizeof(vert), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU, &vert);

	Ilum::Shader::Variant var;
	var.addDefine("Fuck");
	Ilum::Shader shader;

	//auto vert_shader = shader.createShaderModule("D:/Workspace/IlumEngine/Asset/Shader/GLSL/scene_indexing_tes.glsl.vert");
	//auto tesc_shader = shader.createShaderModule("D:/Workspace/IlumEngine/Asset/Shader/GLSL/scene_indexing_tes.glsl.tesc");
	//auto tese_shader = shader.createShaderModule("D:/Workspace/IlumEngine/Asset/Shader/GLSL/scene_indexing_tes.glsl.tese");
	//auto frag_shader = shader.createShaderModule("D:/Workspace/IlumEngine/Asset/Shader/GLSL/scene_indexing_tes.glsl.frag");
	auto vert_shader = shader.createShaderModule("D:/Workspace/IlumEngine/Asset/Shader/GLSL/gbuffer.glsl.vert");
	auto frag_shader = shader.createShaderModule("D:/Workspace/IlumEngine/Asset/Shader/GLSL/gbuffer.glsl.frag");

	auto shader_desc = shader.getShaderDescription();

	Ilum::Attachment back_buffer     = {0, "back_buffer", Ilum::Attachment::Type::Swapchain};
	Ilum::Attachment position_buffer = {1, "position_buffer", Ilum::Attachment::Type::Image, VK_FORMAT_R16G16B16A16_SFLOAT};
	Ilum::Attachment normal_buffer   = {2, "normal_buffer", Ilum::Attachment::Type::Image, VK_FORMAT_R16G16B16A16_SFLOAT};
	Ilum::Attachment albedo_buffer   = {3, "albedo_buffer", Ilum::Attachment::Type::Image, VK_FORMAT_R8G8B8A8_UNORM};
	Ilum::Attachment depth_buffer    = {4, "depth_buffer", Ilum::Attachment::Type::Depth};

	Ilum::RenderTarget render_target(
	    /*Attachment*/ {
	        back_buffer,
	        position_buffer,
	        normal_buffer,
	        albedo_buffer,
	        depth_buffer,
	    },

	    /*Subpass*/ {{0, {0, 1, 2, 3, 4}}, {1, {0, 4}, {1, 2, 3}}, {2, {0, 4}, {1}}},

	    /*Render Area*/
	    {{0, 0}, {100, 100}});

	/*
	const std::vector<std::string> &shader_paths,
	    const RenderTarget &            render_target,
	    PipelineState                   pipeline_state = {},
	    uint32_t                        subpass_index  = 0,
	    const Shader::Variant &         variant        = {});
	*/

	Ilum::PipelineState pso;
	pso.vertex_input_state.attribute_descriptions = {
	    {VkVertexInputAttributeDescription{0, 0, VK_FORMAT_R32G32B32A32_SFLOAT, offsetof(Vertex, pos)}},
	    {VkVertexInputAttributeDescription{1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, color)}},
	    {VkVertexInputAttributeDescription{2, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, normal)}}};

	pso.vertex_input_state.binding_descriptions = {
	    {VkVertexInputBindingDescription{0, sizeof(Vertex), VK_VERTEX_INPUT_RATE_VERTEX}}};

	Ilum::PipelineGraphics pipeline_graphics({"D:/Workspace/IlumEngine/Asset/Shader/GLSL/gbuffer.glsl.vert", "D:/Workspace/IlumEngine/Asset/Shader/GLSL/gbuffer.glsl.frag"}, render_target, pso);

	while (!Ilum::Window::instance()->shouldClose())
	{
		engine->onTick();

		std::this_thread::sleep_for(std::chrono::duration<double, std::milli>(16));

		Ilum::Window::instance()->setTitle(title + " FPS: " + std::to_string(Ilum::Timer::instance()->getFPS()));
	}

	return 0;
}