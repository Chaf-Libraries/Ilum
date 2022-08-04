#include "RHIShader.hpp"

namespace Ilum
{
RHIShader::RHIShader(RHIDevice* device, const std::vector<uint8_t> &source):
    p_device(device)
{
}
}        // namespace Ilum