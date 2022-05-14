#pragma once

#include "../RenderPass.hpp"

#include <Core/Macro.hpp>

#include <rttr/registration>

namespace Ilum
{
class FXAA : public RenderPass
{
  public:
	FXAA();
	~FXAA() = default;

	virtual void Create(RGBuilder &builder) override;
};

RTTR_REGISTRATION
{
	rttr::registration::class_<FXAA>("FXAA")
	    .constructor<>()
	    .method("Create", &FXAA::Create);
}

}        // namespace Ilum