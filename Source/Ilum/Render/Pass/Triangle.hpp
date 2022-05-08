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

	virtual void Create(RGBuilder &builder) override;
};

RTTR_REGISTRATION
{
	rttr::registration::class_<Triangle>("Triangle")
	    .constructor<>()
	    .method("Create", &Triangle::Create);
}

}        // namespace Ilum