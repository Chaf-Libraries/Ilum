#include "CubemapPrefilter.hpp"

namespace Ilum::pass
{
CubemapPrefilter::CubemapPrefilter()
{
}

void CubemapPrefilter::setupPipeline(PipelineState &state)
{
	state.shader.load(std::string(PROJECT_SOURCE_DIR) + "Source/Shaders/PreProcess/CubemapPrefilter.comp", VK_SHADER_STAGE_COMPUTE_BIT, Shader::Type::GLSL);

	state.descriptor_bindings.bind(0, 0, "SkyBox", ImageViewType::Native, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
}

void CubemapPrefilter::resolveResources(ResolveState &resolve)
{

}

void CubemapPrefilter::render(RenderPassState &state)
{

}

void CubemapPrefilter::onImGui()
{

}
}        // namespace Ilum::pass