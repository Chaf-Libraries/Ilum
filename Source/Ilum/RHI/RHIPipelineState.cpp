#include "RHIPipelineState.hpp"

namespace Ilum
{
RHIPipelineState::RHIPipelineState(RHIDevice *device) :
    p_device(device)
{
}

RHIPipelineState &RHIPipelineState::SetShader(RHIShaderStage stage, RHIShader *shader)
{
	if (shader)
	{
		m_shaders[stage] = shader;
	}
	else
	{
		if (m_shaders.find(stage) != m_shaders.end())
		{
			m_shaders.erase(stage);
		}
	}
}
}        // namespace Ilum