#pragma once

#include "../RenderPass.hpp"

#include <Core/Macro.hpp>

#include <rttr/registration>

namespace Ilum
{
class VisualizeVBuffer : public RenderPass
{
  public:
	VisualizeVBuffer();
	~VisualizeVBuffer() = default;

	virtual void Create(RGBuilder &builder) override;
};

RTTR_REGISTRATION
{
	rttr::registration::class_<VisualizeVBuffer>("VisualizeVBuffer")
	    .constructor<>()
	    .method("Create", &VisualizeVBuffer::Create);
}

}        // namespace Ilum