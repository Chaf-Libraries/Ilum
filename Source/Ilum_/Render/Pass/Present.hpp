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

	virtual void Create(RGBuilder &builder) override;

  private:
	bool test = false;
};

RTTR_REGISTRATION
{
	rttr::registration::class_<Present>("Present")
	    .constructor<>()
	    .method("Create", &Present::Create);
}
}        // namespace Ilum