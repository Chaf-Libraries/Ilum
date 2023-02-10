#pragma once

namespace Ilum
{
class RHIContext;
class MaterialGraph;
struct MaterialGraphDesc;

class MaterialGraphBuilder
{
  public:
	MaterialGraphBuilder(RHIContext *rhi_context);

	~MaterialGraphBuilder() = default;

	void Compile(MaterialGraph *graph);

  private:
	RHIContext *p_rhi_context = nullptr;
};
}        // namespace Ilum