#pragma once

#include "Core/Engine/PCH.hpp"

namespace Ilum
{
class RenderTarget;

class RenderPass
{
  public:
	RenderPass(const RenderTarget &render_target);

	~RenderPass();

	operator const VkRenderPass &() const;

	const VkRenderPass &getRenderPass() const;

  private:
	VkRenderPass m_handle = VK_NULL_HANDLE;
};
}        // namespace Ilum