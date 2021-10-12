#include "PipelineState.hpp"

#include "Geometry/Vertex.hpp"

namespace Ilum
{
PipelineState::PipelineState() :
    input_assembly_state({}),
    rasterization_state({}),
    depth_stencil_state({}),
    viewport_state({}),
    multisample_state({}),
    dynamic_state({}),
    vertex_input_state(getVertexInput<Vertex>())
{
}
}        // namespace Ilum