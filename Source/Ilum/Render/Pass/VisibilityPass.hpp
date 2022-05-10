#pragma once

#include "../RenderPass.hpp"

#include <Core/Macro.hpp>

#include <rttr/registration>

namespace Ilum
{
class VisibilityPass : public RenderPass
{
  public:
	VisibilityPass();
	~VisibilityPass() = default;

	virtual void Create(RGBuilder &builder) override;
};

RTTR_REGISTRATION
{
	rttr::registration::class_<VisibilityPass>("VisibilityPass")
	    .constructor<>()
	    .method("Create", &VisibilityPass::Create);
}

}        // namespace Ilum