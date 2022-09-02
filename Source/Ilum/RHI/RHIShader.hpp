#pragma once

#include "RHIDefinitions.hpp"

#include <vector>

namespace Ilum
{
class RHIDevice;

enum class DescriptorType
{
	Sampler,
	TextureSRV,
	TextureUAV,
	ConstantBuffer,
	StructuredBuffer,
	AccelerationStructure
};

struct ShaderMeta
{
	struct Variable
	{
		uint32_t  spirv_id;
		uint32_t  location;
		RHIFormat format;

		inline bool operator==(const Variable &other)
		{
			return location == other.location &&
			       format == other.format;
		}
	};

	struct Constant
	{
		uint32_t       spirv_id;
		std::string    name;
		uint32_t       size   = 0;
		uint32_t       offset = 0;
		RHIShaderStage stage;

		inline bool operator==(const Constant &other)
		{
			return name == other.name &&
			       size == other.size &&
			       offset == other.offset &&
			       stage == other.stage;
		}
	};

	struct Descriptor
	{
		uint32_t       spirv_id;
		std::string    name;
		uint32_t       array_size;
		uint32_t       set;
		uint32_t       binding;
		DescriptorType type;
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
			bool should_add = true;
			for (auto &descriptor : descriptors)
			{
				if (descriptor == rhs_descriptor)
				{
					descriptor.stage = descriptor.stage | rhs_descriptor.stage;
					should_add       = false;
				}
			}
			if (should_add)
			{
				descriptors.push_back(rhs_descriptor);
			}
		}

		for (auto &rhs_constant : rhs.constants)
		{
			bool should_add = true;
			for (auto &constant : constants)
			{
				if (constant == rhs_constant)
				{
					constant.stage = constant.stage | rhs_constant.stage;
				}
			}
			if (should_add)
			{
				constants.push_back(rhs_constant);
			}
		}

		for (auto &input : rhs.inputs)
		{
			inputs.push_back(input);
		}
		for (auto &output : rhs.outputs)
		{
			outputs.push_back(output);
		}

		return *this;
	}

	std::vector<Descriptor> descriptors;
	std::vector<Constant>   constants;
	std::vector<Variable>   inputs;
	std::vector<Variable>   outputs;

	size_t hash = 0;
};

class RHIShader
{
  public:
	RHIShader(RHIDevice *device, const std::string &entry_point, const std::vector<uint8_t> &source);

	virtual ~RHIShader() = default;

	static std::unique_ptr<RHIShader> Create(RHIDevice *device, const std::string &entry_point, const std::vector<uint8_t> &source);

	const std::string &GetEntryPoint() const;

  protected:
	RHIDevice  *p_device = nullptr;
	std::string m_entry_point;
};
}        // namespace Ilum