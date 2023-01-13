#pragma once

#include "../Resource.hpp"

#include <Geometry/Meshlet.hpp>

#define MAX_BONE_INFLUENCE 8

namespace Ilum
{
class RHIContext;
class RHIBuffer;
class RHIAccelerationStructure;

template <>
class EXPORT_API Resource<ResourceType::SkinnedMesh> final : public IResource
{
  public:
	struct SkinnedVertex
	{
		alignas(16) glm::vec3 position;
		alignas(16) glm::vec3 normal;
		alignas(16) glm::vec3 tangent;

		alignas(16) glm::vec2 texcoord0;
		glm::vec2 texcoord1;

		int32_t bones[8]   = {-1, -1, -1, -1, -1, -1, -1, -1};
		float   weights[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

		template<typename Archive>
		void serialize(Archive& archive)
		{
			archive(position, normal, tangent, texcoord0, texcoord1, bones, weights);
		}
	};

  public:
	Resource(RHIContext *rhi_context, const std::string &name);

	Resource(RHIContext *rhi_context, const std::string &name, std::vector<SkinnedVertex> &&vertices, std::vector<uint32_t> &&indices, std::vector<Meshlet> &&meshlets, std::vector<uint32_t> &&meshletdata);

	virtual ~Resource() override;

	virtual bool Validate() const override;

	virtual void Load(RHIContext *rhi_context) override;

	RHIBuffer *GetVertexBuffer() const;

	RHIBuffer *GetIndexBuffer() const;

	RHIBuffer *GetMeshletBuffer() const;

	RHIBuffer *GetMeshletDataBuffer() const;

	size_t GetVertexCount() const;

	size_t GetIndexCount() const;

	size_t GetMeshletCount() const;

	size_t GetBoneCount() const;

	void Update(RHIContext *rhi_context, std::vector<SkinnedVertex> &&vertices, std::vector<uint32_t> &&indices, std::vector<Meshlet> &&meshlets, std::vector<uint32_t> &&meshletdata);

  private:
	struct Impl;
	Impl *m_impl = nullptr;
};
}        // namespace Ilum