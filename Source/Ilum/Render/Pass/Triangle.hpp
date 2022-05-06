#pragma once

#include "../RenderPass.hpp"

#include <Core/Macro.hpp>

#include <rttr/registration>

namespace Ilum
{
class Triangle : public RenderPass
{
  public:
	Triangle();
	~Triangle() = default;

	virtual void Prepare(PipelineState &pso) override;
	virtual void Create(RGBuilder &builder) override;
};

RTTR_REGISTRATION
{
	rttr::registration::class_<Triangle>("Triangle")
	    .constructor<>()
	    .method("Prepare", &Triangle::Prepare)
	    .method("Create", &Triangle::Create);
}

}        // namespace Ilum