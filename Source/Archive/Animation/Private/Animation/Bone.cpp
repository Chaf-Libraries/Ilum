#include "Bone.hpp"

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/quaternion.hpp>

namespace Ilum
{
//struct Bone::Impl
//{
//	std::string name;
//	uint32_t    id;
//	glm::mat4   offset;
//
//	std::vector<KeyPosition> positions;
//	std::vector<KeyRotation> rotations;
//	std::vector<KeyScale>    scales;
//
//	float max_timestamp = 0.f;
//
//	glm::mat4 local_transfrom;
//};
//
//Bone::Bone(
//    const std::string         &name,
//    uint32_t                   id,
//    glm::mat4                  offset,
//    std::vector<KeyPosition> &&positions,
//    std::vector<KeyRotation> &&rotations,
//    std::vector<KeyScale>    &&scales)
//{
//	m_impl = new Impl;
//
//	m_impl->name            = name;
//	m_impl->id              = id;
//	m_impl->offset          = offset;
//	m_impl->positions       = std::move(positions);
//	m_impl->rotations       = std::move(rotations);
//	m_impl->scales          = std::move(scales);
//	m_impl->local_transfrom = glm::mat4(1.f);
//
//	if (!m_impl->positions.empty())
//	{
//		m_impl->max_timestamp = m_impl->positions.back().time_stamp;
//	}
//
//	if (!m_impl->rotations.empty())
//	{
//		m_impl->max_timestamp = m_impl->rotations.back().time_stamp;
//	}
//
//	if (!m_impl->scales.empty())
//	{
//		m_impl->max_timestamp = m_impl->scales.back().time_stamp;
//	}
//}
//
//Bone::Bone(Bone &&bone) noexcept :
//    m_impl(bone.m_impl)
//{
//	bone.m_impl = nullptr;
//}
//
//Bone::~Bone()
//{
//	if (m_impl)
//	{
//		delete m_impl;
//		m_impl = nullptr;
//	}
//}
//
//void Bone::Update(float time)
//{
//	m_impl->local_transfrom =
//	    InterpolatePosition(time) *
//	    InterpolateRotation(time) *
//	    InterpolateScaling(time);
//}
//
//glm::mat4 Bone::GetLocalTransform() const
//{
//	return m_impl->local_transfrom;
//}
//
//std::string Bone::GetBoneName() const
//{
//	return m_impl->name;
//}
//
//uint32_t Bone::GetBoneID() const
//{
//	return m_impl->id;
//}
//
//glm::mat4 Bone::GetBoneOffset() const
//{
//	return m_impl->offset;
//}
//
//size_t Bone::GetPositionIndex(float time) const
//{
//	for (size_t index = 0; index < m_impl->positions.size() - 1; index++)
//	{
//		if (time < m_impl->positions[index + 1].time_stamp)
//		{
//			return index;
//		}
//	}
//	return m_impl->positions.size() - 2;
//}
//
//size_t Bone::GetRotationIndex(float time) const
//{
//	for (size_t index = 0; index < m_impl->rotations.size() - 1; index++)
//	{
//		if (time < m_impl->rotations[index + 1].time_stamp)
//		{
//			return index;
//		}
//	}
//	return m_impl->rotations.size() - 2;
//}
//
//size_t Bone::GetScaleIndex(float time) const
//{
//	for (size_t index = 0; index < m_impl->scales.size() - 1; index++)
//	{
//		if (time < m_impl->scales[index + 1].time_stamp)
//		{
//			return index;
//		}
//	}
//	return m_impl->scales.size() - 2;
//}
//
//glm::mat4 Bone::GetLocalTransform(float time) const
//{
//	glm::mat4 translation = InterpolatePosition(time);
//	glm::mat4 rotation    = InterpolateRotation(time);
//	glm::mat4 scale       = InterpolateScaling(time);
//	return translation * rotation * scale;
//}
//
//glm::mat4 Bone::GetTransformedOffset(float time) const
//{
//	return GetLocalTransform() * m_impl->offset;
//}
//
//float Bone::GetMaxTimeStamp() const
//{
//	return m_impl->max_timestamp;
//}
//
//float Bone::GetScaleFactor(float last, float next, float time) const
//{
//	return (time - last) / (next - last);
//}
//
//glm::mat4 Bone::InterpolatePosition(float time) const
//{
//	if (m_impl->positions.size() == 1)
//	{
//		return glm::translate(glm::mat4(1.f), m_impl->positions[0].position);
//	}
//
//	size_t    p0             = GetPositionIndex(time);
//	size_t    p1             = p0 + 1;
//	float     scale_factor   = GetScaleFactor(m_impl->positions[p0].time_stamp, m_impl->positions[p1].time_stamp, glm::clamp(time, 0.f, m_impl->positions.back().time_stamp));
//	glm::vec3 final_position = glm::mix(m_impl->positions[p0].position, m_impl->positions[p1].position, scale_factor);
//
//	return glm::translate(glm::mat4(1.f), final_position);
//}
//
//glm::mat4 Bone::InterpolateRotation(float time) const
//{
//	if (m_impl->rotations.size() == 1)
//	{
//		auto rotation = glm::normalize(m_impl->rotations[0].orientation);
//		return glm::toMat4(rotation);
//	}
//
//	size_t    p0             = GetRotationIndex(time);
//	size_t    p1             = p0 + 1;
//	float     scale_factor   = GetScaleFactor(m_impl->rotations[p0].time_stamp, m_impl->rotations[p1].time_stamp, glm::clamp(time, 0.f, m_impl->rotations.back().time_stamp));
//	glm::quat final_rotation = glm::slerp(m_impl->rotations[p0].orientation, m_impl->rotations[p1].orientation, scale_factor);
//	final_rotation           = glm::normalize(final_rotation);
//
//	return glm::toMat4(final_rotation);
//}
//
//glm::mat4 Bone::InterpolateScaling(float time) const
//{
//	if (m_impl->scales.size() == 1)
//	{
//		return glm::scale(glm::mat4(1.f), m_impl->scales[0].scale);
//	}
//
//	size_t    p0           = GetScaleIndex(time);
//	size_t    p1           = p0 + 1;
//	float     scale_factor = GetScaleFactor(m_impl->scales[p0].time_stamp, m_impl->scales[p1].time_stamp, glm::clamp(time, 0.f, m_impl->scales.back().time_stamp));
//	glm::vec3 final_scale  = glm::mix(m_impl->scales[p0].scale, m_impl->scales[p1].scale, scale_factor);
//
//	return glm::scale(glm::mat4(1.f), final_scale);
//}
}        // namespace Ilum