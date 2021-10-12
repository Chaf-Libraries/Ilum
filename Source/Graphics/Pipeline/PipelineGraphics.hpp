#pragma once

#include "Pipeline.hpp"
#include "PipelineState.hpp"
#include "Shader.hpp"

namespace Ilum
{
class LogicalDevice;
class RenderTarget;

class PipelineGraphics : public Pipeline
{
  public:
	PipelineGraphics(
	    const std::vector<std::string> &shader_paths,
	    const RenderTarget &            render_target,
	    PipelineState                   pipeline_state = {},
	    uint32_t                        subpass_index  = 0,
	    const Shader::Variant &         variant        = {});

	~PipelineGraphics() = default;

  private:
	PipelineState m_pipeline_state = {};
};
}        // namespace Ilum