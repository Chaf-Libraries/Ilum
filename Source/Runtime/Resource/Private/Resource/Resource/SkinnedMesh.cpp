#include "Resource/SkinnedMesh.hpp"

#include <RHI/RHIContext.hpp>

namespace Ilum
{
struct Resource<ResourceType::SkinnedMesh>::Impl
{
	size_t vertex_count = 0;
	size_t index_count  = 0;
	size_t bone_count   = 0;

	std::unique_ptr<RHIBuffer> vertex_buffer       = nullptr;
	std::unique_ptr<RHIBuffer> index_buffer        = nullptr;
	std::unique_ptr<RHIBuffer> meshlet_buffer      = nullptr;
	std::unique_ptr<RHIBuffer> meshlet_data_buffer = nullptr;
};

Resource<ResourceType::SkinnedMesh>::Resource(RHIContext *rhi_context, const std::string &name, std::vector<SkinnedVertex> &&vertices, std::vector<uint32_t> &&indices, std::vector<Meshlet> &&meshlets, std::vector<uint32_t> &&meshletdata) :
    IResource(name)
{
	m_impl = new Impl;
	Update(rhi_context, std::move(vertices), std::move(indices), std::move(meshlets), std::move(meshletdata));
}

Resource<ResourceType::SkinnedMesh>::~Resource()
{
	delete m_impl;
}

RHIBuffer *Resource<ResourceType::SkinnedMesh>::GetVertexBuffer() const
{
	return m_impl->vertex_buffer.get();
}

RHIBuffer *Resource<ResourceType::SkinnedMesh>::GetIndexBuffer() const
{
	return m_impl->index_buffer.get();
}

size_t Resource<ResourceType::SkinnedMesh>::GetVertexCount() const
{
	return m_impl->vertex_count;
}

size_t Resource<ResourceType::SkinnedMesh>::GetIndexCount() const
{
	return m_impl->index_count;
}

size_t Resource<ResourceType::SkinnedMesh>::GetBoneCount() const
{
	return m_impl->bone_count;
}

void Resource<ResourceType::SkinnedMesh>::Update(RHIContext *rhi_context, std::vector<Resource<ResourceType::SkinnedMesh>::SkinnedVertex> &&vertices, std::vector<uint32_t> &&indices, std::vector<Meshlet> &&meshlets, std::vector<uint32_t> &&meshletdata)
{
	m_impl->vertex_count = vertices.size();
	m_impl->index_count  = indices.size();
	std::unordered_set<int32_t> bone_set;
	for (auto& v : vertices)
	{
		for (uint32_t i = 0; i < 4; i++)
		{
			if (v.bones[i] != -1)
			{
				bone_set.insert(v.bones[i]);
			}
		}
	}
	m_impl->bone_count = bone_set.size();

	m_impl->vertex_buffer       = rhi_context->CreateBuffer<SkinnedVertex>(vertices.size(), RHIBufferUsage::Vertex | RHIBufferUsage::UnorderedAccess | RHIBufferUsage::Transfer, RHIMemoryUsage::GPU_Only);
	m_impl->index_buffer        = rhi_context->CreateBuffer<uint32_t>(indices.size(), RHIBufferUsage::Index | RHIBufferUsage::UnorderedAccess | RHIBufferUsage::Transfer, RHIMemoryUsage::GPU_Only);
	m_impl->meshlet_data_buffer = rhi_context->CreateBuffer<uint32_t>(meshletdata.size(), RHIBufferUsage::UnorderedAccess | RHIBufferUsage::Transfer, RHIMemoryUsage::GPU_Only);
	m_impl->meshlet_buffer      = rhi_context->CreateBuffer<Meshlet>(meshlets.size(), RHIBufferUsage::UnorderedAccess | RHIBufferUsage::Transfer, RHIMemoryUsage::GPU_Only);

	m_impl->vertex_buffer->CopyToDevice(vertices.data(), vertices.size() * sizeof(SkinnedVertex));
	m_impl->index_buffer->CopyToDevice(indices.data(), indices.size() * sizeof(uint32_t));
	m_impl->meshlet_data_buffer->CopyToDevice(meshletdata.data(), meshletdata.size() * sizeof(uint32_t));
	m_impl->meshlet_buffer->CopyToDevice(meshlets.data(), meshlets.size() * sizeof(Meshlet));
}
}        // namespace Ilum