#include "ShaderReflection.hpp"

#include <Core/Hash.hpp>
#include <Core/Macro.hpp>

#include <spirv_glsl.hpp>
#include <spirv_hlsl.hpp>
#include <spirv_reflect.hpp>

namespace Ilum
{
size_t ShaderReflectionData::InputAttachment::Hash() const
{
	size_t seed = 0;
	HashCombine(seed, name);
	HashCombine(seed, array_size);
	HashCombine(seed, input_attachment_index);
	HashCombine(seed, set);
	HashCombine(seed, binding);
	HashCombine(seed, bindless);
	HashCombine(seed, stage);
	return seed;
}

size_t ShaderReflectionData::Image::Hash() const
{
	size_t seed = 0;
	HashCombine(seed, name);
	HashCombine(seed, array_size);
	HashCombine(seed, set);
	HashCombine(seed, binding);
	HashCombine(seed, bindless);
	HashCombine(seed, stage);
	HashCombine(seed, type);
	return seed;
}

size_t ShaderReflectionData::Buffer::Hash() const
{
	size_t seed = 0;
	HashCombine(seed, name);
	HashCombine(seed, size);
	HashCombine(seed, array_size);
	HashCombine(seed, set);
	HashCombine(seed, binding);
	HashCombine(seed, bindless);
	HashCombine(seed, stage);
	return seed;
}

size_t ShaderReflectionData::AccelerationStructure::Hash() const
{
	size_t seed = 0;
	HashCombine(seed, name);
	HashCombine(seed, array_size);
	HashCombine(seed, set);
	HashCombine(seed, binding);
	HashCombine(seed, stage);
	return seed;
}

ShaderReflectionData &ShaderReflectionData::operator+=(const ShaderReflectionData &rhs)
{
	attributes.insert(attributes.end(), rhs.attributes.begin(), rhs.attributes.end());
	input_attachments.insert(input_attachments.end(), rhs.input_attachments.begin(), rhs.input_attachments.end());
	constants.insert(constants.end(), rhs.constants.begin(), rhs.constants.end());

	// Erase duplicate push constant
	std::unordered_set<VkShaderStageFlagBits> stage_set;
	for (auto iter = constants.begin(); iter != constants.end();)
	{
		if (iter->type == ShaderReflectionData::Constant::Type::Push && stage_set.find(iter->stage) != stage_set.end())
		{
			iter = constants.erase(iter);
		}
		else
		{
			stage_set.insert(iter->stage);
			iter++;
		}
	}

	for (auto &image : rhs.images)
	{
		auto iter = std::find_if(images.begin(), images.end(), [image](const Image &img) { return image.binding == img.binding && image.set == img.set; });
		if (iter == images.end())
		{
			images.push_back(image);
		}
		else
		{
			iter->stage |= image.stage;

			// image + sampler combined
			if ((iter->type == Image::Type::Image && image.type == Image::Type::Sampler) ||
			    (iter->type == Image::Type::Sampler && image.type == Image::Type::Image))
			{
				iter->type = Image::Type::ImageSampler;
			}
		}
	}

	for (auto &buffer : rhs.buffers)
	{
		auto iter = std::find_if(buffers.begin(), buffers.end(), [buffer](const Buffer &buf) { return buffer.binding == buf.binding && buffer.set == buf.set; });
		if (iter == buffers.end())
		{
			buffers.push_back(buffer);
		}
		else
		{
			iter->stage |= buffer.stage;
		}
	}

	for (auto &acceleration_structure : rhs.acceleration_structures)
	{
		auto iter = std::find_if(acceleration_structures.begin(), acceleration_structures.end(), [acceleration_structure](const AccelerationStructure &as) { return acceleration_structure.binding == as.binding && acceleration_structure.set == as.set; });
		if (iter == acceleration_structures.end())
		{
			acceleration_structures.push_back(acceleration_structure);
		}
		else
		{
			iter->stage |= acceleration_structure.stage;
		}
	}

	std::for_each(rhs.input_attachments.begin(), rhs.input_attachments.end(), [this](const ShaderReflectionData::InputAttachment &input) { sets.insert(input.set); });
	std::for_each(rhs.images.begin(), rhs.images.end(), [this](const ShaderReflectionData::Image &image) { sets.insert(image.set); });
	std::for_each(rhs.buffers.begin(), rhs.buffers.end(), [this](const ShaderReflectionData::Buffer &buffer) { sets.insert(buffer.set); });
	std::for_each(rhs.acceleration_structures.begin(), rhs.acceleration_structures.end(), [this](const ShaderReflectionData::AccelerationStructure &acceleration_structure) { sets.insert(acceleration_structure.set); });

	return *this;
}

size_t ShaderReflectionData::Hash() const
{
	size_t hash_val = 0;
	for (auto &image : images)
	{
		HashCombine(hash_val, image.Hash());
	}
	for (auto &input : input_attachments)
	{
		HashCombine(hash_val, input.Hash());
	}

	for (auto &buffer : buffers)
	{
		HashCombine(hash_val, buffer.Hash());
	}

	for (auto &acceleration_structure : acceleration_structures)
	{
		HashCombine(hash_val, acceleration_structure.Hash());
	}

	return hash_val;
}

template <spv::Decoration T, typename Descriptor>
inline void read_resource_decoration(const spirv_cross::Compiler &compiler, const spirv_cross::Resource &resource, Descriptor &descriptor)
{
	LOG_ERROR("Not implemented! Read resource decoration of type.");
}

template <>
inline void read_resource_decoration<spv::DecorationLocation, ShaderReflectionData::Attribute>(const spirv_cross::Compiler &compiler, const spirv_cross::Resource &resource, ShaderReflectionData::Attribute &descriptor)
{
	descriptor.location = compiler.get_decoration(resource.id, spv::DecorationLocation);
}

template <>
inline void read_resource_decoration<spv::DecorationDescriptorSet, ShaderReflectionData::Image>(const spirv_cross::Compiler &compiler, const spirv_cross::Resource &resource, ShaderReflectionData::Image &descriptor)
{
	descriptor.set = compiler.get_decoration(resource.id, spv::DecorationDescriptorSet);
}

template <>
inline void read_resource_decoration<spv::DecorationDescriptorSet, ShaderReflectionData::Buffer>(const spirv_cross::Compiler &compiler, const spirv_cross::Resource &resource, ShaderReflectionData::Buffer &descriptor)
{
	descriptor.set = compiler.get_decoration(resource.id, spv::DecorationDescriptorSet);
}

template <>
inline void read_resource_decoration<spv::DecorationDescriptorSet, ShaderReflectionData::AccelerationStructure>(const spirv_cross::Compiler &compiler, const spirv_cross::Resource &resource, ShaderReflectionData::AccelerationStructure &descriptor)
{
	descriptor.set = compiler.get_decoration(resource.id, spv::DecorationDescriptorSet);
}

template <>
inline void read_resource_decoration<spv::DecorationDescriptorSet, ShaderReflectionData::InputAttachment>(const spirv_cross::Compiler &compiler, const spirv_cross::Resource &resource, ShaderReflectionData::InputAttachment &descriptor)
{
	descriptor.set = compiler.get_decoration(resource.id, spv::DecorationDescriptorSet);
}

template <>
inline void read_resource_decoration<spv::DecorationBinding, ShaderReflectionData::InputAttachment>(const spirv_cross::Compiler &compiler, const spirv_cross::Resource &resource, ShaderReflectionData::InputAttachment &descriptor)
{
	descriptor.binding = compiler.get_decoration(resource.id, spv::DecorationBinding);
}

template <>
inline void read_resource_decoration<spv::DecorationBinding, ShaderReflectionData::Image>(const spirv_cross::Compiler &compiler, const spirv_cross::Resource &resource, ShaderReflectionData::Image &descriptor)
{
	descriptor.binding = compiler.get_decoration(resource.id, spv::DecorationBinding);
}

template <>
inline void read_resource_decoration<spv::DecorationBinding, ShaderReflectionData::Buffer>(const spirv_cross::Compiler &compiler, const spirv_cross::Resource &resource, ShaderReflectionData::Buffer &descriptor)
{
	descriptor.binding = compiler.get_decoration(resource.id, spv::DecorationBinding);
}

template <>
inline void read_resource_decoration<spv::DecorationBinding, ShaderReflectionData::AccelerationStructure>(const spirv_cross::Compiler &compiler, const spirv_cross::Resource &resource, ShaderReflectionData::AccelerationStructure &descriptor)
{
	descriptor.binding = compiler.get_decoration(resource.id, spv::DecorationBinding);
}

template <>
inline void read_resource_decoration<spv::DecorationInputAttachmentIndex, ShaderReflectionData::InputAttachment>(const spirv_cross::Compiler &compiler, const spirv_cross::Resource &resource, ShaderReflectionData::InputAttachment &descriptor)
{
	descriptor.input_attachment_index = compiler.get_decoration(resource.id, spv::DecorationInputAttachmentIndex);
}

inline void read_resource_vec_size(const spirv_cross::Compiler &compiler, const spirv_cross::Resource &resource, ShaderReflectionData::Attribute &descriptor)
{
	const auto &spirv_type = compiler.get_type_from_variable(resource.id);

	descriptor.vec_size = spirv_type.vecsize;
	descriptor.columns  = spirv_type.columns;
}

inline void read_resource_array_size(const spirv_cross::Compiler &compiler, const spirv_cross::Resource &resource, ShaderReflectionData::Attribute &descriptor)
{
	const auto &spirv_type = compiler.get_type_from_variable(resource.id);

	descriptor.array_size = spirv_type.array.size() ? spirv_type.array[0] : 1;
}

inline void read_resource_array_size(const spirv_cross::Compiler &compiler, const spirv_cross::Resource &resource, ShaderReflectionData::InputAttachment &descriptor)
{
	const auto &spirv_type = compiler.get_type_from_variable(resource.id);

	descriptor.bindless   = spirv_type.array_size_literal.size() ? spirv_type.array_size_literal[0] : false;
	descriptor.array_size = spirv_type.array.size() ? spirv_type.array[0] : 1;
}

inline void read_resource_array_size(const spirv_cross::Compiler &compiler, const spirv_cross::Resource &resource, ShaderReflectionData::Image &descriptor)
{
	const auto &spirv_type = compiler.get_type_from_variable(resource.id);

	descriptor.bindless   = spirv_type.array_size_literal.size() ? spirv_type.array_size_literal[0] : false;
	descriptor.array_size = spirv_type.array.size() ? spirv_type.array[0] : 1;
}

inline void read_resource_array_size(const spirv_cross::Compiler &compiler, const spirv_cross::Resource &resource, ShaderReflectionData::Buffer &descriptor)
{
	const auto &spirv_type = compiler.get_type_from_variable(resource.id);

	descriptor.bindless   = spirv_type.array_size_literal.size() ? spirv_type.array_size_literal[0] : false;
	descriptor.array_size = spirv_type.array.size() ? spirv_type.array[0] : 1;
}

inline void read_resource_array_size(const spirv_cross::Compiler &compiler, const spirv_cross::Resource &resource, ShaderReflectionData::AccelerationStructure &descriptor)
{
	const auto &spirv_type = compiler.get_type_from_variable(resource.id);

	descriptor.array_size = spirv_type.array.size() ? spirv_type.array[0] : 1;
}

inline void read_resource_size(const spirv_cross::Compiler &compiler, const spirv_cross::Resource &resource, ShaderReflectionData::Buffer &descriptor)
{
	const auto &spirv_type = compiler.get_type_from_variable(resource.id);

	descriptor.size = static_cast<uint32_t>(compiler.get_declared_struct_size_runtime_array(spirv_type, 0));
}

inline void read_resource_size(const spirv_cross::Compiler &compiler, const spirv_cross::Resource &resource, ShaderReflectionData::Constant &descriptor)
{
	const auto &spirv_type = compiler.get_type_from_variable(resource.id);

	descriptor.size = static_cast<uint32_t>(compiler.get_declared_struct_size_runtime_array(spirv_type, 0));
}

inline void read_resource_size(const spirv_cross::Compiler &compiler, const spirv_cross::SPIRConstant &constant, ShaderReflectionData::Constant &descriptor)
{
	auto spirv_type = compiler.get_type(constant.constant_type);

	switch (spirv_type.basetype)
	{
		case spirv_cross::SPIRType::BaseType::Boolean:
		case spirv_cross::SPIRType::BaseType::Char:
		case spirv_cross::SPIRType::BaseType::Int:
		case spirv_cross::SPIRType::BaseType::UInt:
		case spirv_cross::SPIRType::BaseType::Float:
			descriptor.size = 4;
			break;
		case spirv_cross::SPIRType::BaseType::Int64:
		case spirv_cross::SPIRType::BaseType::UInt64:
		case spirv_cross::SPIRType::BaseType::Double:
			descriptor.size = 8;
			break;
		default:
			descriptor.size = 0;
			break;
	}
}

template <typename T>
inline std::vector<T> read_shader_resource(const spirv_cross::Compiler &compiler, VkShaderStageFlagBits stage)
{
	LOG_ERROR("Not implemented! Read shader resources of type.");
}

template <>
inline std::vector<ShaderReflectionData::Attribute> read_shader_resource<ShaderReflectionData::Attribute>(const spirv_cross::Compiler &compiler, VkShaderStageFlagBits stage)
{
	std::vector<ShaderReflectionData::Attribute> attributes;

	// Parsing input attribute
	auto input_resources = compiler.get_shader_resources().stage_inputs;
	for (auto &resource : input_resources)
	{
		ShaderReflectionData::Attribute attribute{};
		attribute.type  = ShaderReflectionData::Attribute::Type::Input;
		attribute.stage = stage;
		attribute.name  = resource.name;

		read_resource_vec_size(compiler, resource, attribute);
		read_resource_array_size(compiler, resource, attribute);
		read_resource_decoration<spv::DecorationLocation>(compiler, resource, attribute);

		attributes.push_back(attribute);
	}

	// Parsing output attribute
	auto output_resources = compiler.get_shader_resources().stage_outputs;
	for (auto &resource : output_resources)
	{
		ShaderReflectionData::Attribute attribute{};
		attribute.type  = ShaderReflectionData::Attribute::Type::Output;
		attribute.stage = stage;
		attribute.name  = resource.name;

		read_resource_vec_size(compiler, resource, attribute);
		read_resource_array_size(compiler, resource, attribute);
		read_resource_decoration<spv::DecorationLocation>(compiler, resource, attribute);

		attributes.push_back(attribute);
	}

	return attributes;
}

template <>
inline std::vector<ShaderReflectionData::InputAttachment> read_shader_resource<ShaderReflectionData::InputAttachment>(const spirv_cross::Compiler &compiler, VkShaderStageFlagBits stage)
{
	std::vector<ShaderReflectionData::InputAttachment> input_attachments;

	auto subpass_resources = compiler.get_shader_resources().subpass_inputs;
	for (auto &resource : subpass_resources)
	{
		ShaderReflectionData::InputAttachment input_attachment{};
		input_attachment.name = resource.name;

		read_resource_array_size(compiler, resource, input_attachment);
		read_resource_decoration<spv::DecorationInputAttachmentIndex>(compiler, resource, input_attachment);
		read_resource_decoration<spv::DecorationDescriptorSet>(compiler, resource, input_attachment);
		read_resource_decoration<spv::DecorationBinding>(compiler, resource, input_attachment);

		input_attachments.push_back(input_attachment);
	}

	return input_attachments;
}

template <>
inline std::vector<ShaderReflectionData::Image> read_shader_resource<ShaderReflectionData::Image>(const spirv_cross::Compiler &compiler, VkShaderStageFlagBits stage)
{
	std::vector<ShaderReflectionData::Image> images;

	// Parsing image
	auto image_resources = compiler.get_shader_resources().separate_images;
	for (auto &resource : image_resources)
	{
		ShaderReflectionData::Image image{};
		image.type  = ShaderReflectionData::Image::Type::Image;
		image.stage = stage;
		image.name  = resource.name;

		read_resource_array_size(compiler, resource, image);
		read_resource_decoration<spv::DecorationDescriptorSet>(compiler, resource, image);
		read_resource_decoration<spv::DecorationBinding>(compiler, resource, image);

		images.push_back(image);
	}

	// Parsing image sampler
	image_resources = compiler.get_shader_resources().sampled_images;
	for (auto &resource : image_resources)
	{
		ShaderReflectionData::Image image{};
		image.type  = ShaderReflectionData::Image::Type::ImageSampler;
		image.stage = stage;
		image.name  = resource.name;

		read_resource_array_size(compiler, resource, image);
		read_resource_decoration<spv::DecorationDescriptorSet>(compiler, resource, image);
		read_resource_decoration<spv::DecorationBinding>(compiler, resource, image);

		images.push_back(image);
	}

	// Parsing image storage
	image_resources = compiler.get_shader_resources().storage_images;
	for (auto &resource : image_resources)
	{
		ShaderReflectionData::Image image{};
		image.type  = ShaderReflectionData::Image::Type::ImageStorage;
		image.stage = stage;
		image.name  = resource.name;

		read_resource_array_size(compiler, resource, image);
		// TODO:
		// read_resource_decoration<spv::DecorationNonReadable>(compiler, resource, image);
		// read_resource_decoration<spv::DecorationNonWritable>(compiler, resource, image);
		read_resource_decoration<spv::DecorationDescriptorSet>(compiler, resource, image);
		read_resource_decoration<spv::DecorationBinding>(compiler, resource, image);

		images.push_back(image);
	}

	// Parsing sampler
	image_resources = compiler.get_shader_resources().separate_samplers;
	for (auto &resource : image_resources)
	{
		ShaderReflectionData::Image image{};
		image.type  = ShaderReflectionData::Image::Type::Sampler;
		image.stage = stage;
		image.name  = resource.name;

		read_resource_array_size(compiler, resource, image);
		read_resource_decoration<spv::DecorationDescriptorSet>(compiler, resource, image);
		read_resource_decoration<spv::DecorationBinding>(compiler, resource, image);

		images.push_back(image);
	}

	return images;
}

template <>
inline std::vector<ShaderReflectionData::Buffer> read_shader_resource<ShaderReflectionData::Buffer>(const spirv_cross::Compiler &compiler, VkShaderStageFlagBits stage)
{
	std::vector<ShaderReflectionData::Buffer> buffers;

	// Parsing uniform buffer
	auto uniform_resources = compiler.get_shader_resources().uniform_buffers;
	for (auto &resource : uniform_resources)
	{
		ShaderReflectionData::Buffer buffer{};
		buffer.type  = ShaderReflectionData::Buffer::Type::Uniform;
		buffer.stage = stage;
		buffer.name  = resource.name;

		read_resource_size(compiler, resource, buffer);
		read_resource_array_size(compiler, resource, buffer);
		read_resource_decoration<spv::DecorationDescriptorSet>(compiler, resource, buffer);
		read_resource_decoration<spv::DecorationBinding>(compiler, resource, buffer);

		buffers.push_back(buffer);
	}

	// Parsing storage buffer
	auto storage_buffer = compiler.get_shader_resources().storage_buffers;
	for (auto &resource : storage_buffer)
	{
		ShaderReflectionData::Buffer buffer{};
		buffer.type  = ShaderReflectionData::Buffer::Type::Storage;
		buffer.stage = stage;
		buffer.name  = resource.name;

		read_resource_size(compiler, resource, buffer);
		read_resource_array_size(compiler, resource, buffer);
		read_resource_decoration<spv::DecorationDescriptorSet>(compiler, resource, buffer);
		read_resource_decoration<spv::DecorationBinding>(compiler, resource, buffer);

		buffers.push_back(buffer);
	}

	return buffers;
}

template <>
inline std::vector<ShaderReflectionData::AccelerationStructure> read_shader_resource<ShaderReflectionData::AccelerationStructure>(const spirv_cross::Compiler &compiler, VkShaderStageFlagBits stage)
{
	std::vector<ShaderReflectionData::AccelerationStructure> acceleration_structures;

	auto shader_acceleration_structures = compiler.get_shader_resources().acceleration_structures;
	for (auto &resource : shader_acceleration_structures)
	{
		ShaderReflectionData::AccelerationStructure acceleration_structure = {};
		acceleration_structure.stage                                       = stage;
		acceleration_structure.name                                        = resource.name;

		read_resource_array_size(compiler, resource, acceleration_structure);
		read_resource_decoration<spv::DecorationDescriptorSet>(compiler, resource, acceleration_structure);
		read_resource_decoration<spv::DecorationBinding>(compiler, resource, acceleration_structure);

		acceleration_structures.push_back(acceleration_structure);
	}

	return acceleration_structures;
}

template <>
inline std::vector<ShaderReflectionData::Constant> read_shader_resource<ShaderReflectionData::Constant>(const spirv_cross::Compiler &compiler, VkShaderStageFlagBits stage)
{
	std::vector<ShaderReflectionData::Constant> constants;

	// Parsing push constant
	auto resources = compiler.get_shader_resources().push_constant_buffers;
	for (auto &resource : resources)
	{
		const auto &spivr_type = compiler.get_type_from_variable(resource.id);

		std::uint32_t offset = std::numeric_limits<std::uint32_t>::max();

		for (auto i = 0U; i < spivr_type.member_types.size(); ++i)
		{
			auto mem_offset = compiler.get_member_decoration(spivr_type.self, i, spv::DecorationOffset);

			offset = std::min(offset, mem_offset);
		}

		ShaderReflectionData::Constant constant{};
		constant.type   = ShaderReflectionData::Constant::Type::Push;
		constant.stage  = stage;
		constant.name   = resource.name;
		constant.offset = offset;

		read_resource_size(compiler, resource, constant);
		constant.size -= constant.offset;

		constants.push_back(constant);
	}

	// Parsing specialization constant
	auto specialization_constants = compiler.get_specialization_constants();
	for (auto &resource : specialization_constants)
	{
		auto &spirv_value = compiler.get_constant(resource.id);

		ShaderReflectionData::Constant constant{};
		constant.type        = ShaderReflectionData::Constant::Type::Specialization;
		constant.stage       = stage;
		constant.name        = compiler.get_name(resource.id);
		constant.offset      = 0;
		constant.constant_id = resource.constant_id;

		read_resource_size(compiler, spirv_value, constant);

		constants.push_back(constant);
	}

	return constants;
}

ShaderReflectionData ShaderReflection::Reflect(const std::vector<uint32_t> &spirv, VkShaderStageFlagBits stage)
{
	ShaderReflectionData data;

	spirv_cross::CompilerReflection compiler(spirv);
	spirv_cross::CompilerGLSL       glsl_compiler(spirv);

	auto opts    = compiler.get_common_options();
	opts.es      = false;
	opts.version = 460;

	glsl_compiler.set_common_options(opts);
	auto test = glsl_compiler.compile();

	opts.vulkan_semantics = true;
	compiler.set_common_options(opts);

	data.stage                   = stage;
	data.attributes              = read_shader_resource<ShaderReflectionData::Attribute>(compiler, stage);
	data.input_attachments       = read_shader_resource<ShaderReflectionData::InputAttachment>(compiler, stage);
	data.images                  = read_shader_resource<ShaderReflectionData::Image>(compiler, stage);
	data.buffers                 = read_shader_resource<ShaderReflectionData::Buffer>(compiler, stage);
	data.acceleration_structures = read_shader_resource<ShaderReflectionData::AccelerationStructure>(compiler, stage);
	data.constants               = read_shader_resource<ShaderReflectionData::Constant>(compiler, stage);

	std::for_each(data.input_attachments.begin(), data.input_attachments.end(), [&data](const ShaderReflectionData::InputAttachment &input) { data.sets.insert(input.set); });
	std::for_each(data.images.begin(), data.images.end(), [&data](const ShaderReflectionData::Image &image) { data.sets.insert(image.set); });
	std::for_each(data.buffers.begin(), data.buffers.end(), [&data](const ShaderReflectionData::Buffer &buffer) { data.sets.insert(buffer.set); });

	for (auto iter = data.images.begin(); iter != data.images.end(); iter++)
	{
		auto next = iter + 1;
		while (next != data.images.end())
		{
			if (iter->set == next->set &&
			    next->binding == iter->binding)
			{
				if ((iter->type == ShaderReflectionData::Image::Type::Image && next->type == ShaderReflectionData::Image::Type::Sampler) ||
				    (iter->type == ShaderReflectionData::Image::Type::Sampler && next->type == ShaderReflectionData::Image::Type::Image))
				{
					iter->type = ShaderReflectionData::Image::Type::ImageSampler;
					next       = data.images.erase(next);
					continue;
				}
			}
			next++;
		}
	}

	return data;
}

}        // namespace Ilum