#pragma once

#include "../RenderPass.hpp"

#include <Core/Macro.hpp>

#include <rttr/registration>

namespace Ilum
{
class SkyboxPass : public RenderPass
{
  public:
	SkyboxPass();
	~SkyboxPass() = default;

	virtual void Create(RGBuilder &builder) override;
};

RTTR_REGISTRATION
{
	rttr::registration::class_<SkyboxPass>("SkyboxPass")
	    .constructor<>()
	    .method("Create", &SkyboxPass::Create);
}

}        // namespace Ilum