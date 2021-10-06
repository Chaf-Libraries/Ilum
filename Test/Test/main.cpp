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
#include <Core/Graphics/GraphicsContext.hpp>
#include <Core/Graphics/Pipeline/Shader.hpp>
#include <Core/Graphics/RenderPass/Swapchain.hpp>

#include <Core/Resource/Bitmap/Bitmap.hpp>

#include <Core/Graphics/Buffer/Buffer.h>
#include <Core/Graphics/Image/Image2D.hpp>
#include <Core/Graphics/Image/ImageDepth.hpp>

#include <Math/Vector2.h>
#include <Math/Vector3.h>
#include <Math/Vector4.h>

struct Vertex
{
	Math::Vector3 pos;
	Math::Vector3 normal;
	Math::Vector2 uv;
	Math::Vector3 color;
	Math::Vector4 tangent;
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

	{
		Ilum::Buffer buffer(sizeof(vert), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU, &vert);

		Ilum::Shader::Variant var;
		var.addDefine("Fuck");
		Ilum::Shader shader;
		shader.setVertexInput<Vertex, uint32_t>();

		auto vert_shader = shader.createShaderModule("D:/Workspace/IlumEngine/Asset/Shader/GLSL/scene_indexing_tes.glsl.vert");
		auto tesc_shader = shader.createShaderModule("D:/Workspace/IlumEngine/Asset/Shader/GLSL/scene_indexing_tes.glsl.tesc");
		auto tese_shader = shader.createShaderModule("D:/Workspace/IlumEngine/Asset/Shader/GLSL/scene_indexing_tes.glsl.tese");
		auto frag_shader = shader.createShaderModule("D:/Workspace/IlumEngine/Asset/Shader/GLSL/scene_indexing_tes.glsl.frag");

		auto shader_desc = shader.createShaderDescription();
	}

	while (!Ilum::Window::instance()->shouldClose())
	{
		engine->onTick();

		std::this_thread::sleep_for(std::chrono::duration<double, std::milli>(16));

		Ilum::Window::instance()->setTitle(title + " FPS: " + std::to_string(Ilum::Timer::instance()->getFPS()));
	}

	return 0;
}