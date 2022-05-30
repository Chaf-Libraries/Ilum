#pragma once

#include "../RenderPass.hpp"

#include <Core/Macro.hpp>

#include <rttr/registration>

namespace Ilum
{
class VisualizeBVH : public RenderPass
{
  public:
	VisualizeBVH();
	~VisualizeBVH() = default;

	virtual void Create(RGBuilder &builder) override;
};

RTTR_REGISTRATION
{
	rttr::registration::class_<VisualizeBVH>("VisualizeBVH")
	    .constructor<>()
	    .method("Create", &VisualizeBVH::Create);
}

}        // namespace Ilum