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

	std::unique_ptr<MaterialGraph> Compile(MaterialGraphDesc &desc);

  private:
	RHIContext *p_rhi_context = nullptr;
};
}        // namespace Ilum