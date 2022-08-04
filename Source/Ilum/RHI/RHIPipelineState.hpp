#pragma once

#include "RHI/RHIDevice.hpp"

namespace Ilum
{
class RHIShader;

struct DepthStencilState
{
	bool depth_test_enable;
	bool depth_write_enable;
	RHICompareOp compare;
	// TODO: Stencil Test
};

struct BlendState
{
	bool enable;
	RHILogicOp logic_op;

	struct AttachmentState
	{
		bool blend_enable;
		
	};

};

class RHIPipelineState
{
  public:
	RHIPipelineState(RHIDevice *device);

	virtual ~RHIPipelineState() = 0;

	RHIPipelineState& SetShader(RHIShaderStage stage, RHIShader *shader);

private:
	RHIDevice *p_device = nullptr;

	std::unordered_map<RHIShaderStage, RHIShader *> m_shaders;
};
}        // namespace Ilum