#include "Resource/Animation.hpp"

#include <RHI/RHIContext.hpp>

namespace Ilum
{
struct Resource<ResourceType::Animation>::Impl
{
	std::vector<Bone> bones;

	std::unique_ptr<RHITexture> skinned_matrics = nullptr;
	std::unique_ptr<RHIBuffer>  bone_matrics    = nullptr;

	HierarchyNode hierarchy;

	float m_max_timestamp = 0.f;

	uint32_t m_bone_count = 0;

	uint32_t frame_count = 0;
};

Bone::Bone(
    const std::string         &name,
    uint32_t                   id,
    glm::mat4                  offset,
    std::vector<KeyPosition> &&positions,
    std::vector<KeyRotation> &&rotations,
    std::vector<KeyScale>    &&scales)
{
	m_name            = name;
	m_id              = id;
	m_offset          = offset;
	m_positions       = std::move(positions);
	m_rotations       = std::move(rotations);
	m_scales          = std::move(scales);
	m_local_transfrom = glm::mat4(1.f);

	if (!m_positions.empty())
	{
		m_max_timestamp = glm::max(m_max_timestamp, m_positions.back().time_stamp);
	}

	if (!m_rotations.empty())
	{
		m_max_timestamp = glm::max(m_max_timestamp, m_rotations.back().time_stamp);
	}

	if (!m_scales.empty())
	{
		m_max_timestamp = glm::max(m_max_timestamp, m_scales.back().time_stamp);
	}
}

void Bone::Update(float time)
{
	m_local_transfrom =
	    InterpolatePosition(time) *
	    InterpolateRotation(time) *
	    InterpolateScaling(time);
}

glm::mat4 Bone::GetLocalTransform() const
{
	return m_local_transfrom;
}

std::string Bone::GetBoneName() const
{
	return m_name;
}

uint32_t Bone::GetBoneID() const
{
	return m_id;
}

glm::mat4 Bone::GetBoneOffset() const
{
	return m_offset;
}

size_t Bone::GetPositionIndex(float time) const
{
	for (size_t index = 0; index < m_positions.size() - 1; index++)
	{
		if (time < m_positions[index + 1].time_stamp)
		{
			return index;
		}
	}
	return m_positions.size() - 2;
}

size_t Bone::GetRotationIndex(float time) const
{
	for (size_t index = 0; index < m_rotations.size() - 1; index++)
	{
		if (time < m_rotations[index + 1].time_stamp)
		{
			return index;
		}
	}
	return m_rotations.size() - 2;
}

size_t Bone::GetScaleIndex(float time) const
{
	for (size_t index = 0; index < m_scales.size() - 1; index++)
	{
		if (time < m_scales[index + 1].time_stamp)
		{
			return index;
		}
	}
	return m_scales.size() - 2;
}

glm::mat4 Bone::GetLocalTransform(float time) const
{
	glm::mat4 translation = InterpolatePosition(time);
	glm::mat4 rotation    = InterpolateRotation(time);
	glm::mat4 scale       = InterpolateScaling(time);
	return translation * rotation * scale;
}

glm::mat4 Bone::GetTransformedOffset(float time) const
{
	return GetLocalTransform() * m_offset;
}

float Bone::GetMaxTimeStamp() const
{
	return m_max_timestamp;
}

float Bone::GetScaleFactor(float last, float next, float time) const
{
	return (time - last) / (next - last);
}

glm::mat4 Bone::InterpolatePosition(float time) const
{
	if (m_positions.empty())
	{
		return glm::mat4(1.f);
	}

	if (m_positions.size() == 1)
	{
		return glm::translate(glm::mat4(1.f), m_positions[0].position);
	}

	size_t    p0             = GetPositionIndex(time);
	size_t    p1             = p0 + 1;
	float     scale_factor   = GetScaleFactor(m_positions[p0].time_stamp, m_positions[p1].time_stamp, glm::clamp(time, 0.f, m_positions.back().time_stamp));
	glm::vec3 final_position = glm::mix(m_positions[p0].position, m_positions[p1].position, scale_factor);

	return glm::translate(glm::mat4(1.f), final_position);
}

glm::mat4 Bone::InterpolateRotation(float time) const
{
	if (m_rotations.empty())
	{
		return glm::mat4(1.f);
	}

	if (m_rotations.size() == 1)
	{
		auto rotation = glm::normalize(m_rotations[0].orientation);
		return glm::toMat4(rotation);
	}

	size_t    p0             = GetRotationIndex(time);
	size_t    p1             = p0 + 1;
	float     scale_factor   = GetScaleFactor(m_rotations[p0].time_stamp, m_rotations[p1].time_stamp, glm::clamp(time, 0.f, m_rotations.back().time_stamp));
	glm::quat final_rotation = glm::slerp(m_rotations[p0].orientation, m_rotations[p1].orientation, scale_factor);
	final_rotation           = glm::normalize(final_rotation);

	return glm::toMat4(final_rotation);
}

glm::mat4 Bone::InterpolateScaling(float time) const
{
	if (m_scales.empty())
	{
		return glm::mat4(1.f);
	}

	if (m_scales.size() == 1)
	{
		return glm::scale(glm::mat4(1.f), m_scales[0].scale);
	}

	size_t    p0           = GetScaleIndex(time);
	size_t    p1           = p0 + 1;
	float     scale_factor = GetScaleFactor(m_scales[p0].time_stamp, m_scales[p1].time_stamp, glm::clamp(time, 0.f, m_scales.back().time_stamp));
	glm::vec3 final_scale  = glm::mix(m_scales[p0].scale, m_scales[p1].scale, scale_factor);

	return glm::scale(glm::mat4(1.f), final_scale);
}

Resource<ResourceType::Animation>::Resource(RHIContext *rhi_context, const std::string &name) :
    IResource(rhi_context, name, ResourceType::Animation)
{
}

Resource<ResourceType::Animation>::Resource(RHIContext *rhi_context, const std::string &name, std::vector<Bone> &&bones, HierarchyNode &&hierarchy) :
    IResource(name)
{
	m_impl = new Impl;

	m_impl->bones     = std::move(bones);
	m_impl->hierarchy = std::move(hierarchy);

	for (auto &bone : m_impl->bones)
	{
		m_impl->m_max_timestamp = glm::max(m_impl->m_max_timestamp, bone.GetMaxTimeStamp());
	}

	std::vector<uint8_t> thumbnail_data;
	DESERIALIZE("Asset/BuildIn/animation_icon.asset", thumbnail_data);
	UpdateThumbnail(rhi_context, thumbnail_data);
	SERIALIZE(fmt::format("Asset/Meta/{}.{}.asset", m_name, (uint32_t) ResourceType::Animation), thumbnail_data, m_impl->bones, m_impl->hierarchy, m_impl->m_max_timestamp);

	Bake(rhi_context);
}

const std::vector<Bone> &Resource<ResourceType::Animation>::GetBones() const
{
	return m_impl->bones;
}

Bone *Resource<ResourceType::Animation>::GetBone(const std::string &name)
{
	auto iter = std::find_if(m_impl->bones.begin(), m_impl->bones.end(), [&](const Bone &bone) { return bone.GetBoneName() == name; });
	return iter == m_impl->bones.end() ? nullptr : &(*iter);
}

uint32_t Resource<ResourceType::Animation>::GetBoneCount() const
{
	return m_impl->m_bone_count;
}

uint32_t Resource<ResourceType::Animation>::GetMaxBoneIndex() const
{
	uint32_t idx = 0;
	for (auto &bone : m_impl->bones)
	{
		idx = std::max(idx, bone.GetBoneID());
	}
	return idx;
}

float Resource<ResourceType::Animation>::GetMaxTimeStamp() const
{
	return m_impl->m_max_timestamp;
}

uint32_t Resource<ResourceType::Animation>::GetFrameCount() const
{
	return static_cast<uint32_t>(m_impl->m_max_timestamp * 30.f);
}

const HierarchyNode &Resource<ResourceType::Animation>::GetHierarchyNode() const
{
	return m_impl->hierarchy;
}

void Resource<ResourceType::Animation>::Bake(RHIContext *rhi_context)
{
	std::function<void(const HierarchyNode &, float time, std::vector<glm::mat4> &, glm::mat4, bool)> calculate_bone_transform = [&](const HierarchyNode &node, float time, std::vector<glm::mat4> &skinned_matrics, glm::mat4 parent, bool outside) {
		Bone *bone = GetBone(node.name);

		glm::mat4 global_transformation = outside ? glm::mat4(1.f) : parent * node.transform;

		if (bone)
		{
			global_transformation    = parent * bone->GetLocalTransform(time);
			uint32_t  bone_id        = bone->GetBoneID();
			glm::mat4 offset         = bone->GetBoneOffset();
			skinned_matrics[bone_id] = global_transformation * offset;

			outside = false;
		}

		for (auto &child : node.children)
		{
			calculate_bone_transform(child, time, skinned_matrics, global_transformation, outside);
		}
	};

	m_impl->m_bone_count = 0;
	for (auto &bone : m_impl->bones)
	{
		m_impl->m_bone_count = glm::max(m_impl->m_bone_count, bone.GetBoneID());
	}
	m_impl->m_bone_count++;

	// Bone matrics
	m_impl->bone_matrics = rhi_context->CreateBuffer(sizeof(glm::mat4) * m_impl->m_bone_count, RHIBufferUsage::ConstantBuffer | RHIBufferUsage::UnorderedAccess, RHIMemoryUsage::CPU_TO_GPU);

	// Skinned Matrics
	// X - bone id
	// Y - frame transform float4x3
	size_t frame_count      = static_cast<size_t>(m_impl->m_max_timestamp * 30.f);
	m_impl->skinned_matrics = rhi_context->CreateTexture2D((uint32_t) m_impl->m_bone_count * 3, (uint32_t) frame_count, RHIFormat::R32G32B32A32_FLOAT, RHITextureUsage::Transfer | RHITextureUsage::ShaderResource, false);

	std::vector<float> skinned_matrics(frame_count * 4ull * 3ull * m_impl->m_bone_count);

	for (size_t i = 0; i < frame_count; i++)
	{
		float time = static_cast<float>(i) / 30.f;

		std::vector<glm::mat4> frame_skinned_matrics(m_impl->m_bone_count);
		calculate_bone_transform(m_impl->hierarchy, time, frame_skinned_matrics, glm::mat4(1.f), true);

		if (i == 0)
		{
			m_impl->bone_matrics->CopyToDevice(frame_skinned_matrics.data(), frame_skinned_matrics.size() * sizeof(glm::mat4));
		}

		size_t offset = 4ull * 3ull * m_impl->m_bone_count * i;

		for (size_t j = 0; j < m_impl->m_bone_count; j++)
		{
			glm::mat4 transform = glm::transpose(frame_skinned_matrics[j]);

			skinned_matrics[offset + 4ull * 3ull * j]      = transform[0][0];
			skinned_matrics[offset + 4ull * 3ull * j + 1]  = transform[0][1];
			skinned_matrics[offset + 4ull * 3ull * j + 2]  = transform[0][2];
			skinned_matrics[offset + 4ull * 3ull * j + 3]  = transform[0][3];
			skinned_matrics[offset + 4ull * 3ull * j + 4]  = transform[1][0];
			skinned_matrics[offset + 4ull * 3ull * j + 5]  = transform[1][1];
			skinned_matrics[offset + 4ull * 3ull * j + 6]  = transform[1][2];
			skinned_matrics[offset + 4ull * 3ull * j + 7]  = transform[1][3];
			skinned_matrics[offset + 4ull * 3ull * j + 8]  = transform[2][0];
			skinned_matrics[offset + 4ull * 3ull * j + 9]  = transform[2][1];
			skinned_matrics[offset + 4ull * 3ull * j + 10] = transform[2][2];
			skinned_matrics[offset + 4ull * 3ull * j + 11] = transform[2][3];
		}
	}

	{
		auto staging_buffer = rhi_context->CreateBuffer(skinned_matrics.size() * sizeof(float), RHIBufferUsage::Transfer, RHIMemoryUsage::CPU_TO_GPU);
		staging_buffer->CopyToDevice(skinned_matrics.data(), skinned_matrics.size() * sizeof(float));
		auto *cmd_buffer = rhi_context->CreateCommand(RHIQueueFamily::Graphics);
		cmd_buffer->Begin();
		cmd_buffer->ResourceStateTransition({TextureStateTransition{m_impl->skinned_matrics.get(), RHIResourceState::Undefined, RHIResourceState::TransferDest}}, {});
		cmd_buffer->CopyBufferToTexture(staging_buffer.get(), m_impl->skinned_matrics.get(), 0, 0, 1);
		cmd_buffer->ResourceStateTransition({TextureStateTransition{m_impl->skinned_matrics.get(), RHIResourceState::TransferDest, RHIResourceState::ShaderResource}}, {});
		cmd_buffer->End();
		rhi_context->Execute(cmd_buffer);
	}
}

RHITexture *Resource<ResourceType::Animation>::GetSkinnedMatrics() const
{
	return m_impl->skinned_matrics.get();
}

RHIBuffer *Resource<ResourceType::Animation>::GetBoneMatrics() const
{
	return m_impl->bone_matrics.get();
}

Resource<ResourceType::Animation>::~Resource()
{
	delete m_impl;
}

bool Resource<ResourceType::Animation>::Validate() const
{
	return m_impl != nullptr;
}

void Resource<ResourceType::Animation>::Load(RHIContext *rhi_context)
{
	m_impl = new Impl;

	std::vector<uint8_t> thumbnail_data;
	DESERIALIZE(fmt::format("Asset/Meta/{}.{}.asset", m_name, (uint32_t) ResourceType::Animation), thumbnail_data, m_impl->bones, m_impl->hierarchy, m_impl->m_max_timestamp);

	Bake(rhi_context);
}
}        // namespace Ilum