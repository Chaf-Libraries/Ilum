#pragma once

#include "../RenderPass.hpp"

#include <Core/Macro.hpp>

#include <rttr/registration>

namespace Ilum
{
class VShading : public RenderPass
{
  public:
	VShading();
	~VShading() = default;

	virtual void Create(RGBuilder &builder) override;
};

RTTR_REGISTRATION
{
	rttr::registration::class_<VShading>("VShading")
	    .constructor<>()
	    .method("Create", &VShading::Create);
}

}        // namespace Ilum