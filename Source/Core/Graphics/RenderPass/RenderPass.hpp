#pragma once

#include "Core/Engine/PCH.hpp"

namespace Ilum
{
class RenderTarget;

class RenderPass
{
  public:
	RenderPass(const RenderTarget &render_target, VkFormat depth_format, VkSampleCountFlagBits samples = VK_SAMPLE_COUNT_1_BIT);

	~RenderPass();

	operator const VkRenderPass &() const;

	const VkRenderPass &getRenderPass() const;

  private:
	VkRenderPass m_handle = VK_NULL_HANDLE;
};
}        // namespace Ilum