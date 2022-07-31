#include <Core/Log.hpp>
#include <Core/Hash.hpp>

#include <RHI/RHIDevice.hpp>

int main()
{
	size_t hash = 0;
	Ilum::HashCombine(hash, 1, 2, 3.f, true);

	auto device = Ilum::RHIDevice::Create();

	LOG_INFO("IsRayTracingSupport: {}", device->IsRayTracingSupport());
	LOG_INFO("IsMeshShaderSupport: {}", device->IsMeshShaderSupport());
	LOG_INFO("IsBufferDeviceAddressSupport: {}", device->IsBufferDeviceAddressSupport());
	LOG_INFO("IsBindlessResourceSupport: {}", device->IsBindlessResourceSupport());

	return 0;
}