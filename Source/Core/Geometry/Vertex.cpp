#include "Vertex.hpp"

namespace Ilum
{
Shader::VertexInput Vertex::getVertexInput()
{
	const std::vector<VkVertexInputBindingDescription> vertex_input_binding = {
	    VkVertexInputBindingDescription{0, sizeof(Vertex), VK_VERTEX_INPUT_RATE_VERTEX}};

	const std::vector<VkVertexInputAttributeDescription> vertex_input_description = {
	    VkVertexInputAttributeDescription{0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, position)},
	    VkVertexInputAttributeDescription{1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, normal)},
	    VkVertexInputAttributeDescription{2, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(Vertex, uv)},
	    VkVertexInputAttributeDescription{3, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, tengent)}};

	return {vertex_input_binding, vertex_input_description};
}

Shader::VertexInput InstanceData::getVertexInput()
{
	const std::vector<VkVertexInputBindingDescription> vertex_input_binding = {
	    VkVertexInputBindingDescription{1, sizeof(InstanceData), VK_VERTEX_INPUT_RATE_INSTANCE}};

	const std::vector<VkVertexInputAttributeDescription> vertex_input_description = {
	    VkVertexInputAttributeDescription{0, 1, VK_FORMAT_R32_UINT, offsetof(InstanceData, index)}};

	return {vertex_input_binding, vertex_input_description};
}
}        // namespace Ilum