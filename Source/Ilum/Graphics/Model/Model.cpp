#include "Model.hpp"

#include "Device/LogicalDevice.hpp"

#include "Graphics/Command/CommandBuffer.hpp"
#include "Graphics/GraphicsContext.hpp"
#include "Graphics/Synchronization/Queue.hpp"
#include "Graphics/Synchronization/QueueSystem.hpp"

#include "Threading/ThreadPool.hpp"

namespace Ilum
{
Model::Model(std::vector<SubMesh> &&submeshes) :
    submeshes(std::move(submeshes))
{
	for (auto& submesh : this->submeshes)
	{
		bounding_box.merge(submesh.bounding_box);
	}

	createBuffer();
}

Model::Model(Model &&other) noexcept :
    submeshes(std::move(other.submeshes)),
    vertex_buffer(std::move(other.vertex_buffer)),
    index_buffer(std::move(other.index_buffer)),
    bounding_box(other.bounding_box)
{
	other.submeshes.clear();
}

Model &Model::operator=(Model &&other) noexcept
{
	submeshes     = std::move(other.submeshes);
	vertex_buffer = std::move(other.vertex_buffer);
	index_buffer  = std::move(other.index_buffer);
	bounding_box  = other.bounding_box;

	other.submeshes.clear();

	return *this;
}

void Model::createBuffer()
{
	if (submeshes.empty())
	{
		return;
	}

	std::vector<Vertex>   vertices;
	std::vector<uint32_t> indices;

	for (const auto &submesh : submeshes)
	{
		uint32_t offset = static_cast<uint32_t>(vertices.size());
		vertices.insert(vertices.end(), submesh.vertices.begin(), submesh.vertices.end());
		std::for_each(submesh.indices.begin(), submesh.indices.end(), [&indices, offset](uint32_t index) { indices.push_back(offset + index); });
	}

	vertex_buffer = Buffer(sizeof(Vertex) * vertices.size(), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
	index_buffer  = Buffer(sizeof(uint32_t) * indices.size(), VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_ONLY);

	// Staging vertex buffer
	{
		Buffer staging_buffer(sizeof(Vertex) * vertices.size(), VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
		auto * data = staging_buffer.map();
		std::memcpy(data, vertices.data(), sizeof(Vertex) * vertices.size());
		staging_buffer.unmap();
		CommandBuffer command_buffer(QueueUsage::Transfer);
		command_buffer.begin();
		command_buffer.copyBuffer(BufferInfo{staging_buffer}, BufferInfo{vertex_buffer}, sizeof(Vertex) * vertices.size());
		command_buffer.end();
		command_buffer.submitIdle();
	}

	// Staging index buffer
	{
		Buffer staging_buffer(sizeof(uint32_t) * indices.size(), VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
		auto * data = staging_buffer.map();
		std::memcpy(data, indices.data(), sizeof(uint32_t) * indices.size());
		staging_buffer.unmap();
		CommandBuffer command_buffer(QueueUsage::Transfer);
		command_buffer.begin();
		command_buffer.copyBuffer(BufferInfo{staging_buffer}, BufferInfo{index_buffer}, sizeof(uint32_t) * indices.size());
		command_buffer.end();
		command_buffer.submitIdle();
	}
}
}        // namespace Ilum