#pragma once

#include <vector>

namespace Ilum
{
class RenderPass;
class RenderGraph;
class RenderGraphBuilder;



class RenderPass
{
  public:
};

class RenderGraph : public RenderPass
{
  public:
  private:
	std::vector<RenderPass *> m_passes;
};

}        // namespace Ilum