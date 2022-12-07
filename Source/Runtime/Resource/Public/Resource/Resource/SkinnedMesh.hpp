#pragma once

#include "../Resource.hpp"

#include <RHI/RHIContext.hpp>

#define MAX_BONE_INFLUENCE 4

namespace Ilum
{
template <>
class EXPORT_API Resource<ResourceType::SkinnedMesh> final : public IResource
{
  public:
	struct SkinnedVertex
	{
		alignas(16) glm::vec3 position;
		alignas(16) glm::vec3 normal;
		alignas(16) glm::vec3 tangent;

		glm::vec2 texcoord0;
		glm::vec2 texcoord1;

		int32_t bones[MAX_BONE_INFLUENCE]   = {-1};
		float   weights[MAX_BONE_INFLUENCE] = {0.f};
	};

  public:
	Resource(RHIContext *rhi_context, const std::string &name, std::vector<SkinnedVertex> &&vertices, std::vector<uint32_t> &&indices);

	virtual ~Resource() override;

  private:
	struct Impl;
	Impl *m_impl = nullptr;
};
}        // namespace Ilum