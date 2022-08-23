#pragma once

#include <RenderCore/RenderGraph/RenderGraph.hpp>

namespace Ilum::Pass
{
class CopyPass
{
  public:
	static RenderPassDesc CreateDesc(size_t &handle);
};

RENDER_PASS_REGISTERATION(CopyPass);
}        // namespace Ilum::Pass