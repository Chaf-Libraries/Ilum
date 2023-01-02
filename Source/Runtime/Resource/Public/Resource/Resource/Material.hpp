#pragma once

#include "../Resource.hpp"

#include <RHI/RHIContext.hpp>

namespace Ilum
{
class MaterialGraphDesc;
struct MaterialCompilationContext;

template <>
class EXPORT_API Resource<ResourceType::Material> final : public IResource
{
  public:
	Resource(RHIContext *rhi_context, const std::string &name);

	Resource(RHIContext *rhi_context, const std::string &name, MaterialGraphDesc &&desc);

	virtual ~Resource() override;

	virtual bool Validate() const override;

	virtual void Load(RHIContext *rhi_context) override;

	void Compile(const std::string &layout = "");

	const std::string &GetLayout() const;

	MaterialGraphDesc &GetDesc();

  private:
	struct Impl;
	Impl *m_impl = nullptr;
};
}        // namespace Ilum