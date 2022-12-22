#pragma once

#include "../Resource.hpp"

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/quaternion.hpp>

namespace Ilum
{
class Animation;
class RHIContext;
class RHIBuffer;
class RHITexture;

class EXPORT_API Bone
{
  public:
	struct KeyPosition
	{
		glm::vec3 position;
		float     time_stamp;
	};

	struct KeyRotation
	{
		glm::quat orientation;
		float     time_stamp;
	};

	struct KeyScale
	{
		glm::vec3 scale;
		float     time_stamp;
	};

	struct BoneMatrix
	{
		float     frame;
		glm::mat4 transform = glm::mat4(1.f);
	};

  public:
	Bone(
	    const std::string         &name,
	    uint32_t                   id,
	    glm::mat4                  offset,
	    std::vector<KeyPosition> &&positions,
	    std::vector<KeyRotation> &&rotations,
	    std::vector<KeyScale>    &&scales);

	~Bone();

	Bone(const Bone &) = delete;

	Bone &operator=(const Bone &) = delete;

	Bone(Bone &&bone) noexcept;

	Bone &operator=(Bone &&) = delete;

	void Update(float time);

	glm::mat4 GetLocalTransform() const;

	std::string GetBoneName() const;

	uint32_t GetBoneID() const;

	glm::mat4 GetBoneOffset() const;

	size_t GetPositionIndex(float time) const;

	size_t GetRotationIndex(float time) const;

	size_t GetScaleIndex(float time) const;

	glm::mat4 GetLocalTransform(float time) const;

	glm::mat4 GetTransformedOffset(float time) const;

	float GetMaxTimeStamp() const;

	size_t GetFrameCount() const;

  private:
	float GetScaleFactor(float last, float next, float time) const;

	glm::mat4 InterpolatePosition(float time) const;

	glm::mat4 InterpolateRotation(float time) const;

	glm::mat4 InterpolateScaling(float time) const;

  private:
	struct Impl;
	Impl *m_impl = nullptr;
};

struct HierarchyNode
{
	std::string                name;
	glm::mat4                  transform;
	std::vector<HierarchyNode> children;
};

template <>
class EXPORT_API Resource<ResourceType::Animation> final : public IResource
{
  public:
	Resource(RHIContext *rhi_context, const std::string &name, std::vector<Bone> &&bones, HierarchyNode &&hierarchy, float duration, float ticks_per_sec);

	virtual ~Resource() override;

	const std::vector<Bone> &GetBones() const;

	Bone *GetBone(const std::string &name);

	uint32_t GetBoneCount() const;

	uint32_t GetMaxBoneIndex() const;

	float GetMaxTimeStamp() const;

	uint32_t GetFrameCount() const;

	const HierarchyNode &GetHierarchyNode() const;

	void Bake(RHIContext *rhi_context);

	RHITexture *GetSkinnedMatrics() const;

	RHIBuffer *GetBoneMatrics() const;

  private:
	struct Impl;
	Impl *m_impl = nullptr;
};
}        // namespace Ilum