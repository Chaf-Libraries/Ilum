#include "Core/Graphics/Pipeline/PipelineState.hpp"
#include "Vertex.hpp"

namespace Ilum
{
template <>
VertexInputState getVertexInput<Vertex>()
{
	VertexInputState vertex_input_state;

	vertex_input_state.binding_descriptions = {
	    VkVertexInputBindingDescription{0, sizeof(Vertex), VK_VERTEX_INPUT_RATE_VERTEX}};

	vertex_input_state.attribute_descriptions = {
	    VkVertexInputAttributeDescription{0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, position)},
	    VkVertexInputAttributeDescription{1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, normal)},
	    VkVertexInputAttributeDescription{2, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(Vertex, uv)},
	    VkVertexInputAttributeDescription{3, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, tengent)}};

	return vertex_input_state;
}

template <>
VertexInputState getVertexInput<InstanceData>()
{
	VertexInputState vertex_input_state;

	vertex_input_state.binding_descriptions = {
	    VkVertexInputBindingDescription{0, sizeof(InstanceData), VK_VERTEX_INPUT_RATE_INSTANCE}};

	vertex_input_state.attribute_descriptions = {
	    VkVertexInputAttributeDescription{0, 0, VK_FORMAT_R32_UINT, offsetof(InstanceData, index)}};

	return vertex_input_state;
}

template <>
VertexInputState getVertexInput<Vertex, InstanceData>()
{
	VertexInputState vertex_input_state;

	vertex_input_state.binding_descriptions = {
	    VkVertexInputBindingDescription{0, sizeof(Vertex), VK_VERTEX_INPUT_RATE_VERTEX},
	    VkVertexInputBindingDescription{1, sizeof(InstanceData), VK_VERTEX_INPUT_RATE_INSTANCE}};

	vertex_input_state.attribute_descriptions = {
	    VkVertexInputAttributeDescription{0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, position)},
	    VkVertexInputAttributeDescription{1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, normal)},
	    VkVertexInputAttributeDescription{2, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(Vertex, uv)},
	    VkVertexInputAttributeDescription{3, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, tengent)},
	    VkVertexInputAttributeDescription{0, 1, VK_FORMAT_R32_UINT, offsetof(InstanceData, index)}};

	return vertex_input_state;
}
}        // namespace Ilum