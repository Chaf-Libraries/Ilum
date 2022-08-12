#include "RHI/RHIDescriptor.hpp"

namespace Ilum
{
RHIDescriptor::RHIDescriptor(RHIDevice *device, const ShaderMeta &meta):
    p_device(device), m_meta(meta)
{
}
}        // namespace Ilum