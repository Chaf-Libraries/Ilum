#pragma once

#include "RHIDefinitions.hpp"

#include <vector>

namespace Ilum
{
class RHIDevice;

struct ShaderMeta
{
	struct Descriptor
	{
		enum class Type
		{
			Sampler,
			TextureSRV,
			TextureUAV,
			ConstantBuffer,
			StructuredBuffer,
			AccelerationStructure
		};

		std::string    name;
		uint32_t       array_size;
		uint32_t       set;
		uint32_t       binding;
		Type           type;
		RHIShaderStage stage;

		inline bool operator==(const Descriptor &other)
		{
			return name == other.name &&
			       array_size == other.array_size &&
			       set == other.set &&
			       binding == other.binding &&
			       type == other.type;
		}
	};

	inline ShaderMeta &operator+=(const ShaderMeta &rhs)
	{
		for (auto &rhs_descriptor : rhs.descriptors)
		{
			for (auto& descriptor : descriptors)
			{
				if (descriptor == rhs_descriptor)
				{
					descriptor.stage = descriptor.stage | rhs_descriptor.stage;
				}
				else
				{
					descriptors.push_back(rhs_descriptor);
				}
			}
		}
		return *this;
	}

	std::vector<Descriptor> descriptors;

	size_t hash = 0;
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