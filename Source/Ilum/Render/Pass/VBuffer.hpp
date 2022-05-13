#pragma once

#include "../RenderPass.hpp"

#include <Core/Macro.hpp>

#include <rttr/registration>

namespace Ilum
{
class VBuffer : public RenderPass
{
  public:
	VBuffer();
	~VBuffer() = default;

	virtual void Create(RGBuilder &builder) override;
};

RTTR_REGISTRATION
{
	rttr::registration::class_<VBuffer>("VBuffer")
	    .constructor<>()
	    .method("Create", &VBuffer::Create);
}

}        // namespace Ilum