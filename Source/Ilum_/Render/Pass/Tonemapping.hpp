#pragma once

#include "../RenderPass.hpp"

#include <Core/Macro.hpp>

#include <rttr/registration>

namespace Ilum
{
class Tonemapping : public RenderPass
{
  public:
	Tonemapping();
	~Tonemapping() = default;

	virtual void Create(RGBuilder &builder) override;

	private:
	
};

RTTR_REGISTRATION
{
	rttr::registration::class_<Tonemapping>("Tonemapping")
	    .constructor<>()
	    .method("Create", &Tonemapping::Create);
}

}        // namespace Ilum