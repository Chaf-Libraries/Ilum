#pragma once

#include "../Resource.hpp"

namespace Ilum
{
class Animation;

template <>
class EXPORT_API Resource<ResourceType::Animation> final : public IResource
{
  public:
	Resource(Animation &&animation);

	virtual ~Resource() override;

	const Animation &GetAnimation() const;

  private:
	struct Impl;
	Impl *m_impl = nullptr;
};
}        // namespace Ilum