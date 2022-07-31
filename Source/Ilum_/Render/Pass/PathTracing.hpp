#pragma once

#include "../RenderPass.hpp"

#include <Core/Macro.hpp>

#include <rttr/registration>

namespace Ilum
{
class PathTracing : public RenderPass
{
  public:
	PathTracing();
	~PathTracing() = default;

	virtual void Create(RGBuilder &builder) override;
};

RTTR_REGISTRATION
{
	rttr::registration::class_<PathTracing>("PathTracing")
	    .constructor<>()
	    .method("Create", &PathTracing::Create);
}

}        // namespace Ilum