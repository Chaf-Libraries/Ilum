#pragma once

#include "../RenderPass.hpp"

#include <Core/Macro.hpp>

#include <rttr/registration>

namespace Ilum
{
class Present : public RenderPass
{
  public:
	Present();
	~Present() = default;

	virtual void Prepare(PipelineState &pso) override;
	virtual void Create(RGBuilder &builder) override;
};

RTTR_REGISTRATION
{
	rttr::registration::class_<Present>("Present")
	    .constructor<>()
	    .method("Prepare", &Present::Prepare)
	    .method("Create", &Present::Create);
}
}