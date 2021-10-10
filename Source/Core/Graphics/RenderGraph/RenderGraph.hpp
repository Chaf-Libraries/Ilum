#pragma once

#include "Core/Engine/PCH.hpp"

namespace Ilum
{
class RenderTarget;

class PassNode
{
  public:
	PassNode(const ref<RenderTarget> &render_target, uint32_t subpass_index = 0);

  private:
	uint32_t             m_subpass_index;
	std::set<PassNode *> m_prev_passes;
	std::set<PassNode *> m_next_passes;
};

class RenderGraph
{
  public:


  private:
	std::vector<PassNode> m_passes;
};
}        // namespace Ilum