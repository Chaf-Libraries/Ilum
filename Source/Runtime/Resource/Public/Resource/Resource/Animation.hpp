#pragma once

#include "../Resource.hpp"

#include <Animation/Bone.hpp>

namespace Ilum
{
class Animation;
class RHIContext;

template <>
class EXPORT_API Resource<ResourceType::Animation> final : public IResource
{
  public:
	Resource(RHIContext *rhi_context, const std::string &name, std::vector<Bone> &&bones, std::map<std::string, std::pair<glm::mat4, std::string>> &&hierarchy, float duration, float ticks_per_sec);

	virtual ~Resource() override;

	const std::vector<Bone> &GetBones() const;

	Bone *GetBone(const std::string &name);

	uint32_t GetBoneCount() const;

	uint32_t GetMaxBoneIndex() const;

	float GetMaxTimeStamp() const;

	const std::map<std::string, std::pair<glm::mat4, std::string>> &GetHierarchy() const;

  private:
	struct Impl;
	Impl *m_impl = nullptr;
};
}        // namespace Ilum