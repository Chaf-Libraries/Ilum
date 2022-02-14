#include "SpirvReflection.hpp"

#include <Core/Hash.hpp>

#include <spirv_glsl.hpp>

namespace std
{
template <>
struct hash<Ilum::Graphics::ReflectionData::InputAttachment>
{
	size_t operator()(const Ilum::Graphics::ReflectionData::InputAttachment &input_attachment) const
	{
		size_t seed = 0;
		Ilum::Core::HashCombine(seed, input_attachment.name);
		Ilum::Core::HashCombine(seed, input_attachment.array_size);
		Ilum::Core::HashCombine(seed, input_attachment.input_attachment_index);
		Ilum::Core::HashCombine(seed, input_attachment.set);
		Ilum::Core::HashCombine(seed, input_attachment.binding);
		Ilum::Core::HashCombine(seed, input_attachment.bindless);
		Ilum::Core::HashCombine(seed, input_attachment.stage);
		return seed;
	}
};

template <>
struct hash<Ilum::Graphics::ReflectionData::Image>
{
	size_t operator()(const Ilum::Graphics::ReflectionData::Image &image) const
	{
		size_t seed = 0;
		Ilum::Core::HashCombine(seed, image.name);
		Ilum::Core::HashCombine(seed, image.array_size);
		Ilum::Core::HashCombine(seed, image.set);
		Ilum::Core::HashCombine(seed, image.binding);
		Ilum::Core::HashCombine(seed, image.bindless);
		Ilum::Core::HashCombine(seed, image.stage);
		Ilum::Core::HashCombine(seed, image.type);
		return seed;
	}
};

template <>
struct hash<Ilum::Graphics::ReflectionData::Buffer>
{
	size_t operator()(const Ilum::Graphics::ReflectionData::Buffer &buffer) const
	{
		size_t seed = 0;
		Ilum::Core::HashCombine(seed, buffer.name);
		Ilum::Core::HashCombine(seed, buffer.size);
		Ilum::Core::HashCombine(seed, buffer.array_size);
		Ilum::Core::HashCombine(seed, buffer.set);
		Ilum::Core::HashCombine(seed, buffer.binding);
		Ilum::Core::HashCombine(seed, buffer.bindless);
		Ilum::Core::HashCombine(seed, buffer.stage);
		return seed;
	}
};
}        // namespace std

namespace Ilum::Graphics
{
ReflectionData &ReflectionData::operator+=(const ReflectionData &rhs)
{
	attributes.insert(attributes.end(), rhs.attributes.begin(), rhs.attributes.end());
	input_attachments.insert(input_attachments.end(), rhs.input_attachments.begin(), rhs.input_attachments.end());
	constants.insert(constants.end(), rhs.constants.begin(), rhs.constants.end());

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
			iter->qualifiers = iter->qualifiers | image.qualifiers;
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
			iter->qualifiers = iter->qualifiers | buffer.qualifiers;
		}
	}

	std::for_each(rhs.input_attachments.begin(), rhs.input_attachments.end(), [this](const ReflectionData::InputAttachment &input) { sets.insert(input.set); });
	std::for_each(rhs.images.begin(), rhs.images.end(), [this](const ReflectionData::Image &image) { sets.insert(image.set); });
	std::for_each(rhs.buffers.begin(), rhs.buffers.end(), [this](const ReflectionData::Buffer &buffer) { sets.insert(buffer.set); });

	UpdateHash();

	return *this;
}

void ReflectionData::UpdateHash()
{
	hash = 0;
	for (auto &image : images)
	{
		Ilum::Core::HashCombine(hash, image);
	}
	for (auto &input : input_attachments)
	{
		Ilum::Core::HashCombine(hash, input);
	}
	for (auto &buffer : buffers)
	{
		Ilum::Core::HashCombine(hash, buffer);
	}
}

template <spv::Decoration T, typename Descriptor>
inline void ReadResourceDecoration(const spirv_cross::Compiler &compiler, const spirv_cross::Resource &resource, Descriptor &descriptor)
{
	LOG_ERROR("Not implemented! Read resource decoration of type.");
}

template <>
inline void ReadResourceDecoration<spv::DecorationLocation, ReflectionData::Attribute>(const spirv_cross::Compiler &compiler, const spirv_cross::Resource &resource, ReflectionData::Attribute &descriptor)
{
	descriptor.location = compiler.get_decoration(resource.id, spv::DecorationLocation);
}

template <>
inline void ReadResourceDecoration<spv::DecorationDescriptorSet, ReflectionData::Image>(const spirv_cross::Compiler &compiler, const spirv_cross::Resource &resource, ReflectionData::Image &descriptor)
{
	descriptor.set = compiler.get_decoration(resource.id, spv::DecorationDescriptorSet);
}

template <>
inline void ReadResourceDecoration<spv::DecorationDescriptorSet, ReflectionData::Buffer>(const spirv_cross::Compiler &compiler, const spirv_cross::Resource &resource, ReflectionData::Buffer &descriptor)
{
	descriptor.set = compiler.get_decoration(resource.id, spv::DecorationDescriptorSet);
}

template <>
inline void ReadResourceDecoration<spv::DecorationDescriptorSet, ReflectionData::InputAttachment>(const spirv_cross::Compiler &compiler, const spirv_cross::Resource &resource, ReflectionData::InputAttachment &descriptor)
{
	descriptor.set = compiler.get_decoration(resource.id, spv::DecorationDescriptorSet);
}

template <>
inline void ReadResourceDecoration<spv::DecorationBinding, ReflectionData::InputAttachment>(const spirv_cross::Compiler &compiler, const spirv_cross::Resource &resource, ReflectionData::InputAttachment &descriptor)
{
	descriptor.binding = compiler.get_decoration(resource.id, spv::DecorationBinding);
}

template <>
inline void ReadResourceDecoration<spv::DecorationBinding, ReflectionData::Image>(const spirv_cross::Compiler &compiler, const spirv_cross::Resource &resource, ReflectionData::Image &descriptor)
{
	descriptor.binding = compiler.get_decoration(resource.id, spv::DecorationBinding);
}

template <>
inline void ReadResourceDecoration<spv::DecorationBinding, ReflectionData::Buffer>(const spirv_cross::Compiler &compiler, const spirv_cross::Resource &resource, ReflectionData::Buffer &descriptor)
{
	descriptor.binding = compiler.get_decoration(resource.id, spv::DecorationBinding);
}

template <>
inline void ReadResourceDecoration<spv::DecorationNonReadable, ReflectionData::Image>(const spirv_cross::Compiler &compiler, const spirv_cross::Resource &resource, ReflectionData::Image &descriptor)
{
	descriptor.qualifiers = descriptor.qualifiers | ShaderResourceQualifiers::NonReadable;
}

template <>
inline void ReadResourceDecoration<spv::DecorationNonReadable, ReflectionData::Buffer>(const spirv_cross::Compiler &compiler, const spirv_cross::Resource &resource, ReflectionData::Buffer &descriptor)
{
	descriptor.qualifiers = descriptor.qualifiers | ShaderResourceQualifiers::NonReadable;
}

template <>
inline void ReadResourceDecoration<spv::DecorationNonWritable, ReflectionData::Image>(const spirv_cross::Compiler &compiler, const spirv_cross::Resource &resource, ReflectionData::Image &descriptor)
{
	descriptor.qualifiers = descriptor.qualifiers | ShaderResourceQualifiers::NonWriteable;
}

template <>
inline void ReadResourceDecoration<spv::DecorationNonWritable, ReflectionData::Buffer>(const spirv_cross::Compiler &compiler, const spirv_cross::Resource &resource, ReflectionData::Buffer &descriptor)
{
	descriptor.qualifiers = descriptor.qualifiers | ShaderResourceQualifiers::NonWriteable;
}

template <>
inline void ReadResourceDecoration<spv::DecorationInputAttachmentIndex, ReflectionData::InputAttachment>(const spirv_cross::Compiler &compiler, const spirv_cross::Resource &resource, ReflectionData::InputAttachment &descriptor)
{
	descriptor.input_attachment_index = compiler.get_decoration(resource.id, spv::DecorationInputAttachmentIndex);
}

inline void ReadResourceVecSize(const spirv_cross::Compiler &compiler, const spirv_cross::Resource &resource, ReflectionData::Attribute &descriptor)
{
	const auto &spirv_type = compiler.get_type_from_variable(resource.id);

	descriptor.vec_size = spirv_type.vecsize;
	descriptor.columns  = spirv_type.columns;
}

inline void ReadResourceArraySize(const spirv_cross::Compiler &compiler, const spirv_cross::Resource &resource, ReflectionData::Attribute &descriptor)
{
	const auto &spirv_type = compiler.get_type_from_variable(resource.id);

	descriptor.array_size = spirv_type.array.size() ? spirv_type.array[0] : 1;
}

inline void ReadResourceArraySize(const spirv_cross::Compiler &compiler, const spirv_cross::Resource &resource, ReflectionData::InputAttachment &descriptor)
{
	const auto &spirv_type = compiler.get_type_from_variable(resource.id);

	descriptor.bindless   = spirv_type.array_size_literal.size() ? spirv_type.array_size_literal[0] : false;
	descriptor.array_size = spirv_type.array.size() ? spirv_type.array[0] : 1;
}

inline void ReadResourceArraySize(const spirv_cross::Compiler &compiler, const spirv_cross::Resource &resource, ReflectionData::Image &descriptor)
{
	const auto &spirv_type = compiler.get_type_from_variable(resource.id);

	descriptor.bindless   = spirv_type.array_size_literal.size() ? spirv_type.array_size_literal[0] : false;
	descriptor.array_size = spirv_type.array.size() ? spirv_type.array[0] : 1;
}

inline void ReadResourceArraySize(const spirv_cross::Compiler &compiler, const spirv_cross::Resource &resource, ReflectionData::Buffer &descriptor)
{
	const auto &spirv_type = compiler.get_type_from_variable(resource.id);

	descriptor.bindless   = spirv_type.array_size_literal.size() ? spirv_type.array_size_literal[0] : false;
	descriptor.array_size = spirv_type.array.size() ? spirv_type.array[0] : 1;
}

inline void ReadResourceSize(const spirv_cross::Compiler &compiler, const spirv_cross::Resource &resource, ReflectionData::Buffer &descriptor)
{
	const auto &spirv_type = compiler.get_type_from_variable(resource.id);

	descriptor.size = static_cast<uint32_t>(compiler.get_declared_struct_size_runtime_array(spirv_type, 0));
}

inline void ReadResourceSize(const spirv_cross::Compiler &compiler, const spirv_cross::Resource &resource, ReflectionData::Constant &descriptor)
{
	const auto &spirv_type = compiler.get_type_from_variable(resource.id);

	descriptor.size = static_cast<uint32_t>(compiler.get_declared_struct_size_runtime_array(spirv_type, 0));
}

inline void ReadResourceSize(const spirv_cross::Compiler &compiler, const spirv_cross::SPIRConstant &constant, ReflectionData::Constant &descriptor)
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
inline std::vector<T> ReadShaderResource(const spirv_cross::Compiler &compiler, VkShaderStageFlagBits stage)
{
	LOG_ERROR("Not implemented! Read shader resources of type.");
}

template <>
inline std::vector<ReflectionData::Attribute> ReadShaderResource<ReflectionData::Attribute>(const spirv_cross::Compiler &compiler, VkShaderStageFlagBits stage)
{
	std::vector<ReflectionData::Attribute> attributes;

	// Parsing input attribute
	auto input_resources = compiler.get_shader_resources().stage_inputs;
	for (auto &resource : input_resources)
	{
		ReflectionData::Attribute attribute{};
		attribute.type  = ReflectionData::Attribute::Type::Input;
		attribute.stage = stage;
		attribute.name  = resource.name;

		ReadResourceVecSize(compiler, resource, attribute);
		ReadResourceArraySize(compiler, resource, attribute);
		ReadResourceDecoration<spv::DecorationLocation>(compiler, resource, attribute);

		attributes.push_back(attribute);
	}

	// Parsing output attribute
	auto output_resources = compiler.get_shader_resources().stage_outputs;
	for (auto &resource : output_resources)
	{
		ReflectionData::Attribute attribute{};
		attribute.type  = ReflectionData::Attribute::Type::Output;
		attribute.stage = stage;
		attribute.name  = resource.name;

		ReadResourceVecSize(compiler, resource, attribute);
		ReadResourceArraySize(compiler, resource, attribute);
		ReadResourceDecoration<spv::DecorationLocation>(compiler, resource, attribute);

		attributes.push_back(attribute);
	}

	return attributes;
}

template <>
inline std::vector<ReflectionData::InputAttachment> ReadShaderResource<ReflectionData::InputAttachment>(const spirv_cross::Compiler &compiler, VkShaderStageFlagBits stage)
{
	std::vector<ReflectionData::InputAttachment> input_attachments;

	auto subpass_resources = compiler.get_shader_resources().subpass_inputs;
	for (auto &resource : subpass_resources)
	{
		ReflectionData::InputAttachment input_attachment{};
		input_attachment.name = resource.name;

		ReadResourceArraySize(compiler, resource, input_attachment);
		ReadResourceDecoration<spv::DecorationInputAttachmentIndex>(compiler, resource, input_attachment);
		ReadResourceDecoration<spv::DecorationDescriptorSet>(compiler, resource, input_attachment);
		ReadResourceDecoration<spv::DecorationBinding>(compiler, resource, input_attachment);

		input_attachments.push_back(input_attachment);
	}

	return input_attachments;
}

template <>
inline std::vector<ReflectionData::Image> ReadShaderResource<ReflectionData::Image>(const spirv_cross::Compiler &compiler, VkShaderStageFlagBits stage)
{
	std::vector<ReflectionData::Image> images;

	// Parsing image
	auto image_resources = compiler.get_shader_resources().separate_images;
	for (auto &resource : image_resources)
	{
		ReflectionData::Image image{};
		image.type  = ReflectionData::Image::Type::Image;
		image.stage = stage;
		image.name  = resource.name;

		ReadResourceArraySize(compiler, resource, image);
		ReadResourceDecoration<spv::DecorationDescriptorSet>(compiler, resource, image);
		ReadResourceDecoration<spv::DecorationBinding>(compiler, resource, image);

		images.push_back(image);
	}

	// Parsing image sampler
	image_resources = compiler.get_shader_resources().sampled_images;
	for (auto &resource : image_resources)
	{
		ReflectionData::Image image{};
		image.type  = ReflectionData::Image::Type::ImageSampler;
		image.stage = stage;
		image.name  = resource.name;

		ReadResourceArraySize(compiler, resource, image);
		ReadResourceDecoration<spv::DecorationDescriptorSet>(compiler, resource, image);
		ReadResourceDecoration<spv::DecorationBinding>(compiler, resource, image);

		images.push_back(image);
	}

	// Parsing image storage
	image_resources = compiler.get_shader_resources().storage_images;
	for (auto &resource : image_resources)
	{
		ReflectionData::Image image{};
		image.type  = ReflectionData::Image::Type::ImageStorage;
		image.stage = stage;
		image.name  = resource.name;

		ReadResourceArraySize(compiler, resource, image);

		ReadResourceDecoration<spv::DecorationNonReadable>(compiler, resource, image);
		ReadResourceDecoration<spv::DecorationNonWritable>(compiler, resource, image);
		ReadResourceDecoration<spv::DecorationDescriptorSet>(compiler, resource, image);
		ReadResourceDecoration<spv::DecorationBinding>(compiler, resource, image);

		images.push_back(image);
	}

	// Parsing sampler
	image_resources = compiler.get_shader_resources().separate_samplers;
	for (auto &resource : image_resources)
	{
		ReflectionData::Image image{};
		image.type  = ReflectionData::Image::Type::Sampler;
		image.stage = stage;
		image.name  = resource.name;

		ReadResourceArraySize(compiler, resource, image);
		ReadResourceDecoration<spv::DecorationDescriptorSet>(compiler, resource, image);
		ReadResourceDecoration<spv::DecorationBinding>(compiler, resource, image);

		images.push_back(image);
	}

	return images;
}

template <>
inline std::vector<ReflectionData::Buffer> ReadShaderResource<ReflectionData::Buffer>(const spirv_cross::Compiler &compiler, VkShaderStageFlagBits stage)
{
	std::vector<ReflectionData::Buffer> buffers;

	// Parsing uniform buffer
	auto uniform_resources = compiler.get_shader_resources().uniform_buffers;
	for (auto &resource : uniform_resources)
	{
		ReflectionData::Buffer buffer{};
		buffer.type  = ReflectionData::Buffer::Type::Uniform;
		buffer.stage = stage;
		buffer.name  = resource.name;

		ReadResourceSize(compiler, resource, buffer);
		ReadResourceArraySize(compiler, resource, buffer);
		ReadResourceDecoration<spv::DecorationDescriptorSet>(compiler, resource, buffer);
		ReadResourceDecoration<spv::DecorationBinding>(compiler, resource, buffer);

		buffers.push_back(buffer);
	}

	// Parsing storage buffer
	auto storage_buffer = compiler.get_shader_resources().storage_buffers;
	for (auto &resource : storage_buffer)
	{
		ReflectionData::Buffer buffer{};
		buffer.type  = ReflectionData::Buffer::Type::Storage;
		buffer.stage = stage;
		buffer.name  = resource.name;

		ReadResourceSize(compiler, resource, buffer);
		ReadResourceArraySize(compiler, resource, buffer);
		ReadResourceDecoration<spv::DecorationNonReadable>(compiler, resource, buffer);
		ReadResourceDecoration<spv::DecorationNonWritable>(compiler, resource, buffer);
		ReadResourceDecoration<spv::DecorationDescriptorSet>(compiler, resource, buffer);
		ReadResourceDecoration<spv::DecorationBinding>(compiler, resource, buffer);

		buffers.push_back(buffer);
	}

	return buffers;
}

template <>
inline std::vector<ReflectionData::Constant> ReadShaderResource<ReflectionData::Constant>(const spirv_cross::Compiler &compiler, VkShaderStageFlagBits stage)
{
	std::vector<ReflectionData::Constant> constants;

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

		ReflectionData::Constant constant{};
		constant.type   = ReflectionData::Constant::Type::Push;
		constant.stage  = stage;
		constant.name   = resource.name;
		constant.offset = offset;

		ReadResourceSize(compiler, resource, constant);
		constant.size -= constant.offset;

		constants.push_back(constant);
	}

	// Parsing specialization constant
	auto specialization_constants = compiler.get_specialization_constants();
	for (auto &resource : specialization_constants)
	{
		auto &spirv_value = compiler.get_constant(resource.id);

		ReflectionData::Constant constant{};
		constant.type        = ReflectionData::Constant::Type::Specialization;
		constant.stage       = stage;
		constant.name        = compiler.get_name(resource.id);
		constant.offset      = 0;
		constant.constant_id = resource.constant_id;

		ReadResourceSize(compiler, spirv_value, constant);

		constants.push_back(constant);
	}

	return constants;
}

ReflectionData SpirvReflection::Reflect(const std::vector<uint32_t> &spirv, VkShaderStageFlagBits stage)
{
	ReflectionData data;

	spirv_cross::CompilerGLSL compiler(spirv);

	auto opts                     = compiler.get_common_options();
	opts.enable_420pack_extension = true;

	compiler.set_common_options(opts);

	data.stage             = stage;
	data.attributes        = ReadShaderResource<ReflectionData::Attribute>(compiler, stage);
	data.input_attachments = ReadShaderResource<ReflectionData::InputAttachment>(compiler, stage);
	data.images            = ReadShaderResource<ReflectionData::Image>(compiler, stage);
	data.buffers           = ReadShaderResource<ReflectionData::Buffer>(compiler, stage);
	data.constants         = ReadShaderResource<ReflectionData::Constant>(compiler, stage);

	std::for_each(data.input_attachments.begin(), data.input_attachments.end(), [&data](const ReflectionData::InputAttachment &input) { data.sets.insert(input.set); });
	std::for_each(data.images.begin(), data.images.end(), [&data](const ReflectionData::Image &image) { data.sets.insert(image.set); });
	std::for_each(data.buffers.begin(), data.buffers.end(), [&data](const ReflectionData::Buffer &buffer) { data.sets.insert(buffer.set); });

	return data;
}
}        // namespace Ilum::Graphics