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

#include <Math/Vector3.h>
#include <Math/Vector2.h>
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
	Ilum::Engine engine;

	Ilum::Bitmap bitmap("../Asset/Texture/613934.jpg");

	auto *window = engine.getContext().getSubsystem<Ilum::Window>();
	auto *timer  = engine.getContext().getSubsystem<Ilum::Timer>();

	const std::string title = window->getTitle();

	auto *graphics_context = engine.getContext().getSubsystem<Ilum::GraphicsContext>();

	Ilum::Shader::Variant var;
	var.addDefine("Fuck");
	Ilum::Shader shader(graphics_context->getLogicalDevice());
	shader.setVertexInput<Vertex, uint32_t>();

	auto  vert_shader      = shader.createShaderModule("D:/Workspace/IlumEngine/Asset/Shader/GLSL/scene_indexing_tes.glsl.vert");
	auto  tesc_shader      = shader.createShaderModule("D:/Workspace/IlumEngine/Asset/Shader/GLSL/scene_indexing_tes.glsl.tesc");
	auto  tese_shader      = shader.createShaderModule("D:/Workspace/IlumEngine/Asset/Shader/GLSL/scene_indexing_tes.glsl.tese");
	auto  frag_shader      = shader.createShaderModule("D:/Workspace/IlumEngine/Asset/Shader/GLSL/scene_indexing_tes.glsl.frag");

	auto  shader_desc      = shader.createShaderDescription();

	while (!window->shouldClose())
	{
		engine.onTick();

		std::this_thread::sleep_for(std::chrono::duration<double, std::milli>(16));

		window->setTitle(title + " FPS: " + std::to_string(timer->getFPS()));
	}

	return 0;
}