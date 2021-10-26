#include "Model.hpp"

#include "Graphics/Command/CommandBuffer.hpp"

namespace Ilum
{
Model::Model(std::vector<SubMesh> &&submeshes) :
    m_submeshes(std::move(submeshes))
{
	createBuffer();
}

Model::Model(Model &&other) noexcept :
    m_submeshes(std::move(other.m_submeshes))
{
	other.m_submeshes.clear();

	createBuffer();
}

Model &Model::operator=(Model &&other) noexcept
{
	m_submeshes = std::move(other.m_submeshes);

	other.m_submeshes.clear();

	createBuffer();

	return *this;
}

const std::vector<SubMesh> &Model::getSubMeshes() const
{
	return m_submeshes;
}

BufferReference Model::getVertexBuffer() const
{
	return m_vertex_buffer;
}

BufferReference Model::getIndexBuffer() const
{
	return m_index_buffer;
}

void Model::createBuffer()
{
	if (m_submeshes.empty())
	{
		return;
	}

	std::vector<Vertex>   vertices;
	std::vector<uint32_t> indices;

	for (const auto &submesh : m_submeshes)
	{
		uint32_t offset = static_cast<uint32_t>(vertices.size());
		vertices.insert(vertices.end(), submesh.getVertices().begin(), submesh.getVertices().end());
		std::for_each(submesh.getIndices().begin(), submesh.getIndices().end(), [&indices, offset](uint32_t index) { indices.push_back(offset + index); });
	}

	m_vertex_buffer = Buffer(sizeof(Vertex) * vertices.size(), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
	m_index_buffer  = Buffer(sizeof(uint32_t) * indices.size(), VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_ONLY);

	// Staging vertex buffer
	{
		Buffer staging_buffer(sizeof(Vertex) * vertices.size(), VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
		auto * data = staging_buffer.map();
		std::memcpy(data, vertices.data(), sizeof(Vertex) * vertices.size());
		staging_buffer.unmap();
		CommandBuffer command_buffer;
		command_buffer.begin();
		command_buffer.copyBuffer(BufferInfo{staging_buffer}, BufferInfo{m_vertex_buffer}, sizeof(Vertex) * vertices.size());
		command_buffer.end();
		command_buffer.submitIdle();
	}

	// Staging index buffer
	{
		Buffer staging_buffer(sizeof(uint32_t) * indices.size(), VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
		auto * data = staging_buffer.map();
		std::memcpy(data, indices.data(), sizeof(uint32_t) * indices.size());
		staging_buffer.unmap();
		CommandBuffer command_buffer;
		command_buffer.begin();
		command_buffer.copyBuffer(BufferInfo{staging_buffer}, BufferInfo{m_index_buffer}, sizeof(uint32_t) * indices.size());
		command_buffer.end();
		command_buffer.submitIdle();
	}
}
}        // namespace Ilum