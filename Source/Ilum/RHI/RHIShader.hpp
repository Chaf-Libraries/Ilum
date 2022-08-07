#pragma once

#include "RHIDefinitions.hpp"

#include <vector>

namespace Ilum
{
class RHIDevice;

struct ShaderMeta
{
	struct Attribute
	{
		enum class Type
		{
			None,
			Input,
			Output
		};

		std::string name;
		uint32_t    location;
		Type        type;
		RHIShaderStage stage;
	};

	struct Texture
	{
		enum class Type
		{
			SRV,
			UAV
		};
	};
};

class RHIShader
{
  public:
	RHIShader(RHIDevice *device, const std::vector<uint8_t> &source);

	virtual ~RHIShader() = 0;

  protected:
	RHIDevice *p_device = nullptr;
};
}        // namespace Ilum