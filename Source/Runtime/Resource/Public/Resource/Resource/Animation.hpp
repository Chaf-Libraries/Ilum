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

class Bone
{
  public:
	struct KeyPosition
	{
		glm::vec3 position;
		float     time_stamp;

		template <typename Archive>
		void serialize(Archive &archive)
		{
			archive(position, time_stamp);
		}
	};

	struct KeyRotation
	{
		glm::quat orientation;
		float     time_stamp;

		template <typename Archive>
		void serialize(Archive &archive)
		{
			archive(orientation, time_stamp);
		}
	};

	struct KeyScale
	{
		glm::vec3 scale;
		float     time_stamp;

		template <typename Archive>
		void serialize(Archive &archive)
		{
			archive(scale, time_stamp);
		}
	};

	struct BoneMatrix
	{
		float     frame;
		glm::mat4 transform = glm::mat4(1.f);

		template <typename Archive>
		void serialize(Archive &archive)
		{
			archive(frame, transform);
		}
	};

  public:
	Bone() = default;

	Bone(
	    const std::string         &name,
	    uint32_t                   id,
	    glm::mat4                  offset,
	    std::vector<KeyPosition> &&positions,
	    std::vector<KeyRotation> &&rotations,
	    std::vector<KeyScale>    &&scales);

	~Bone() = default;

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

	template <typename Archive>
	void serialize(Archive &archive)
	{
		archive(m_name, m_id, m_offset, m_positions, m_rotations, m_scales, m_max_timestamp, m_local_transfrom);
	}

  private:
	float GetScaleFactor(float last, float next, float time) const;

	glm::mat4 InterpolatePosition(float time) const;

	glm::mat4 InterpolateRotation(float time) const;

	glm::mat4 InterpolateScaling(float time) const;

  private:
	std::string m_name;
	uint32_t    m_id;
	glm::mat4   m_offset;

	std::vector<KeyPosition> m_positions;
	std::vector<KeyRotation> m_rotations;
	std::vector<KeyScale>    m_scales;

	float m_max_timestamp = 0.f;

	glm::mat4 m_local_transfrom;
};

struct HierarchyNode
{
	std::string name;
	glm::mat4   transform;

	std::vector<HierarchyNode> children;

	template <typename Archive>
	void serialize(Archive &archive)
	{
		archive(name, transform, children);
	}
};

template <>
class Resource<ResourceType::Animation> final : public IResource
{
  public:
	Resource(RHIContext *rhi_context, const std::string &name);

	Resource(RHIContext *rhi_context, const std::string &name, std::vector<Bone> &&bones, HierarchyNode &&hierarchy);

	virtual ~Resource() override;

	virtual bool Validate() const override;

	virtual void Load(RHIContext *rhi_context) override;

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