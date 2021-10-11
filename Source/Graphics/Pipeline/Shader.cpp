#include "Shader.hpp"

#include "Device/LogicalDevice.hpp"
#include "Device/PhysicalDevice.hpp"

#include "Engine/Context.hpp"
#include "Engine/Engine.hpp"

#include "File/FileSystem.hpp"

#include "Graphics/GraphicsContext.hpp"

#include <glslang/Include/ResourceLimits.h>
#include <glslang/SPIRV/GLSL.std.450.h>
#include <glslang/SPIRV/GlslangToSpv.h>

#include <spirv_glsl.hpp>

namespace glslang
{
const TBuiltInResource DefaultTBuiltInResource = {
    /* .MaxLights = */ 32,
    /* .MaxClipPlanes = */ 6,
    /* .MaxTextureUnits = */ 32,
    /* .MaxTextureCoords = */ 32,
    /* .MaxVertexAttribs = */ 64,
    /* .MaxVertexUniformComponents = */ 4096,
    /* .MaxVaryingFloats = */ 64,
    /* .MaxVertexTextureImageUnits = */ 32,
    /* .MaxCombinedTextureImageUnits = */ 80,
    /* .MaxTextureImageUnits = */ 32,
    /* .MaxFragmentUniformComponents = */ 4096,
    /* .MaxDrawBuffers = */ 32,
    /* .MaxVertexUniformVectors = */ 128,
    /* .MaxVaryingVectors = */ 8,
    /* .MaxFragmentUniformVectors = */ 16,
    /* .MaxVertexOutputVectors = */ 16,
    /* .MaxFragmentInputVectors = */ 15,
    /* .MinProgramTexelOffset = */ -8,
    /* .MaxProgramTexelOffset = */ 7,
    /* .MaxClipDistances = */ 8,
    /* .MaxComputeWorkGroupCountX = */ 65535,
    /* .MaxComputeWorkGroupCountY = */ 65535,
    /* .MaxComputeWorkGroupCountZ = */ 65535,
    /* .MaxComputeWorkGroupSizeX = */ 1024,
    /* .MaxComputeWorkGroupSizeY = */ 1024,
    /* .MaxComputeWorkGroupSizeZ = */ 64,
    /* .MaxComputeUniformComponents = */ 1024,
    /* .MaxComputeTextureImageUnits = */ 16,
    /* .MaxComputeImageUniforms = */ 8,
    /* .MaxComputeAtomicCounters = */ 8,
    /* .MaxComputeAtomicCounterBuffers = */ 1,
    /* .MaxVaryingComponents = */ 60,
    /* .MaxVertexOutputComponents = */ 64,
    /* .MaxGeometryInputComponents = */ 64,
    /* .MaxGeometryOutputComponents = */ 128,
    /* .MaxFragmentInputComponents = */ 128,
    /* .MaxImageUnits = */ 8,
    /* .MaxCombinedImageUnitsAndFragmentOutputs = */ 8,
    /* .MaxCombinedShaderOutputResources = */ 8,
    /* .MaxImageSamples = */ 0,
    /* .MaxVertexImageUniforms = */ 0,
    /* .MaxTessControlImageUniforms = */ 0,
    /* .MaxTessEvaluationImageUniforms = */ 0,
    /* .MaxGeometryImageUniforms = */ 0,
    /* .MaxFragmentImageUniforms = */ 8,
    /* .MaxCombinedImageUniforms = */ 8,
    /* .MaxGeometryTextureImageUnits = */ 16,
    /* .MaxGeometryOutputVertices = */ 256,
    /* .MaxGeometryTotalOutputComponents = */ 1024,
    /* .MaxGeometryUniformComponents = */ 1024,
    /* .MaxGeometryVaryingComponents = */ 64,
    /* .MaxTessControlInputComponents = */ 128,
    /* .MaxTessControlOutputComponents = */ 128,
    /* .MaxTessControlTextureImageUnits = */ 16,
    /* .MaxTessControlUniformComponents = */ 1024,
    /* .MaxTessControlTotalOutputComponents = */ 4096,
    /* .MaxTessEvaluationInputComponents = */ 128,
    /* .MaxTessEvaluationOutputComponents = */ 128,
    /* .MaxTessEvaluationTextureImageUnits = */ 16,
    /* .MaxTessEvaluationUniformComponents = */ 1024,
    /* .MaxTessPatchComponents = */ 120,
    /* .MaxPatchVertices = */ 32,
    /* .MaxTessGenLevel = */ 64,
    /* .MaxViewports = */ 16,
    /* .MaxVertexAtomicCounters = */ 0,
    /* .MaxTessControlAtomicCounters = */ 0,
    /* .MaxTessEvaluationAtomicCounters = */ 0,
    /* .MaxGeometryAtomicCounters = */ 0,
    /* .MaxFragmentAtomicCounters = */ 8,
    /* .MaxCombinedAtomicCounters = */ 8,
    /* .MaxAtomicCounterBindings = */ 1,
    /* .MaxVertexAtomicCounterBuffers = */ 0,
    /* .MaxTessControlAtomicCounterBuffers = */ 0,
    /* .MaxTessEvaluationAtomicCounterBuffers = */ 0,
    /* .MaxGeometryAtomicCounterBuffers = */ 0,
    /* .MaxFragmentAtomicCounterBuffers = */ 1,
    /* .MaxCombinedAtomicCounterBuffers = */ 1,
    /* .MaxAtomicCounterBufferSize = */ 16384,
    /* .MaxTransformFeedbackBuffers = */ 4,
    /* .MaxTransformFeedbackInterleavedComponents = */ 64,
    /* .MaxCullDistances = */ 8,
    /* .MaxCombinedClipAndCullDistances = */ 8,
    /* .MaxSamples = */ 4,
    /* .maxMeshOutputVerticesNV = */ 256,
    /* .maxMeshOutputPrimitivesNV = */ 512,
    /* .maxMeshWorkGroupSizeX_NV = */ 32,
    /* .maxMeshWorkGroupSizeY_NV = */ 1,
    /* .maxMeshWorkGroupSizeZ_NV = */ 1,
    /* .maxTaskWorkGroupSizeX_NV = */ 32,
    /* .maxTaskWorkGroupSizeY_NV = */ 1,
    /* .maxTaskWorkGroupSizeZ_NV = */ 1,
    /* .maxMeshViewCountNV = */ 4,
    /* .maxDualSourceDrawBuffersEXT = */ 1,

    /* .limits = */ {
        /* .nonInductiveForLoops = */ 1,
        /* .whileLoops = */ 1,
        /* .doWhileLoops = */ 1,
        /* .generalUniformIndexing = */ 1,
        /* .generalAttributeMatrixVectorIndexing = */ 1,
        /* .generalVaryingIndexing = */ 1,
        /* .generalSamplerIndexing = */ 1,
        /* .generalVariableIndexing = */ 1,
        /* .generalConstantMatrixVectorIndexing = */ 1,
    }};
}

namespace Ilum
{
inline EShLanguage get_shader_language(VkShaderStageFlags stage)
{
	switch (stage)
	{
		case VK_SHADER_STAGE_VERTEX_BIT:
			return EShLangVertex;
		case VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT:
			return EShLangTessControl;
		case VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT:
			return EShLangTessEvaluation;
		case VK_SHADER_STAGE_GEOMETRY_BIT:
			return EShLangGeometry;
		case VK_SHADER_STAGE_FRAGMENT_BIT:
			return EShLangFragment;
		case VK_SHADER_STAGE_COMPUTE_BIT:
			return EShLangCompute;
		case VK_SHADER_STAGE_RAYGEN_BIT_KHR:
			return EShLangRayGen;
		case VK_SHADER_STAGE_ANY_HIT_BIT_KHR:
			return EShLangAnyHit;
		case VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR:
			return EShLangClosestHit;
		case VK_SHADER_STAGE_MISS_BIT_KHR:
			return EShLangMiss;
		case VK_SHADER_STAGE_INTERSECTION_BIT_KHR:
			return EShLangIntersect;
		case VK_SHADER_STAGE_CALLABLE_BIT_KHR:
			return EShLangCallable;
		case VK_SHADER_STAGE_MESH_BIT_NV:
			return EShLangMeshNV;
		case VK_SHADER_STAGE_TASK_BIT_NV:
			return EShLangTaskNV;
		default:
			return EShLangCount;
	}
}

template <spv::Decoration T, typename Descriptor>
inline void read_resource_decoration(const spirv_cross::Compiler &compiler, const spirv_cross::Resource &resource, Descriptor &descriptor)
{
	LOG_ERROR("Not implemented! Read resource decoration of type.");
}

template <>
inline void read_resource_decoration<spv::DecorationLocation, Shader::Attribute>(const spirv_cross::Compiler &compiler, const spirv_cross::Resource &resource, Shader::Attribute &descriptor)
{
	descriptor.location = compiler.get_decoration(resource.id, spv::DecorationLocation);
}

template <>
inline void read_resource_decoration<spv::DecorationDescriptorSet, Shader::Image>(const spirv_cross::Compiler &compiler, const spirv_cross::Resource &resource, Shader::Image &descriptor)
{
	descriptor.set = compiler.get_decoration(resource.id, spv::DecorationDescriptorSet);
}

template <>
inline void read_resource_decoration<spv::DecorationDescriptorSet, Shader::Buffer>(const spirv_cross::Compiler &compiler, const spirv_cross::Resource &resource, Shader::Buffer &descriptor)
{
	descriptor.set = compiler.get_decoration(resource.id, spv::DecorationDescriptorSet);
}

template <>
inline void read_resource_decoration<spv::DecorationDescriptorSet, Shader::InputAttachment>(const spirv_cross::Compiler &compiler, const spirv_cross::Resource &resource, Shader::InputAttachment &descriptor)
{
	descriptor.set = compiler.get_decoration(resource.id, spv::DecorationDescriptorSet);
}

template <>
inline void read_resource_decoration<spv::DecorationBinding, Shader::InputAttachment>(const spirv_cross::Compiler &compiler, const spirv_cross::Resource &resource, Shader::InputAttachment &descriptor)
{
	descriptor.binding = compiler.get_decoration(resource.id, spv::DecorationBinding);
}

template <>
inline void read_resource_decoration<spv::DecorationBinding, Shader::Image>(const spirv_cross::Compiler &compiler, const spirv_cross::Resource &resource, Shader::Image &descriptor)
{
	descriptor.binding = compiler.get_decoration(resource.id, spv::DecorationBinding);
}

template <>
inline void read_resource_decoration<spv::DecorationBinding, Shader::Buffer>(const spirv_cross::Compiler &compiler, const spirv_cross::Resource &resource, Shader::Buffer &descriptor)
{
	descriptor.binding = compiler.get_decoration(resource.id, spv::DecorationBinding);
}

template <>
inline void read_resource_decoration<spv::DecorationInputAttachmentIndex, Shader::InputAttachment>(const spirv_cross::Compiler &compiler, const spirv_cross::Resource &resource, Shader::InputAttachment &descriptor)
{
	descriptor.input_attachment_index = compiler.get_decoration(resource.id, spv::DecorationInputAttachmentIndex);
}

template <>
inline void read_resource_decoration<spv::DecorationNonWritable, Shader::Image>(const spirv_cross::Compiler &compiler, const spirv_cross::Resource &resource, Shader::Image &descriptor)
{
	descriptor.qualifiers |= Shader::ShaderResourceQualifiers::NonWritable;
}

template <>
inline void read_resource_decoration<spv::DecorationNonWritable, Shader::Buffer>(const spirv_cross::Compiler &compiler, const spirv_cross::Resource &resource, Shader::Buffer &descriptor)
{
	descriptor.qualifiers |= Shader::ShaderResourceQualifiers::NonWritable;
}

template <>
inline void read_resource_decoration<spv::DecorationNonReadable, Shader::Image>(const spirv_cross::Compiler &compiler, const spirv_cross::Resource &resource, Shader::Image &descriptor)
{
	descriptor.qualifiers |= Shader::ShaderResourceQualifiers::NonReadable;
}

template <>
inline void read_resource_decoration<spv::DecorationNonReadable, Shader::Buffer>(const spirv_cross::Compiler &compiler, const spirv_cross::Resource &resource, Shader::Buffer &descriptor)
{
	descriptor.qualifiers |= Shader::ShaderResourceQualifiers::NonReadable;
}

inline void read_resource_vec_size(const spirv_cross::Compiler &compiler, const spirv_cross::Resource &resource, Shader::Attribute &descriptor)
{
	const auto &spirv_type = compiler.get_type_from_variable(resource.id);

	descriptor.vec_size  = spirv_type.vecsize;
	descriptor.columns   = spirv_type.columns;
	descriptor.base_type = spirv_type.basetype;
}

inline void read_resource_array_size(const spirv_cross::Compiler &compiler, const spirv_cross::Resource &resource, Shader::Attribute &descriptor)
{
	const auto &spirv_type = compiler.get_type_from_variable(resource.id);

	descriptor.array_size = spirv_type.array.size() ? spirv_type.array[0] : 1;
}

inline void read_resource_array_size(const spirv_cross::Compiler &compiler, const spirv_cross::Resource &resource, Shader::InputAttachment &descriptor)
{
	const auto &spirv_type = compiler.get_type_from_variable(resource.id);

	descriptor.array_size = spirv_type.array.size() ? spirv_type.array[0] : 1;
}

inline void read_resource_array_size(const spirv_cross::Compiler &compiler, const spirv_cross::Resource &resource, Shader::Image &descriptor)
{
	const auto &spirv_type = compiler.get_type_from_variable(resource.id);

	descriptor.array_size = spirv_type.array.size() ? spirv_type.array[0] : 1;
}

inline void read_resource_array_size(const spirv_cross::Compiler &compiler, const spirv_cross::Resource &resource, Shader::Buffer &descriptor)
{
	const auto &spirv_type = compiler.get_type_from_variable(resource.id);

	descriptor.array_size = spirv_type.array.size() ? spirv_type.array[0] : 1;
}

inline void read_resource_size(const spirv_cross::Compiler &compiler, const spirv_cross::Resource &resource, Shader::Buffer &descriptor)
{
	const auto &spirv_type = compiler.get_type_from_variable(resource.id);

	descriptor.size = static_cast<uint32_t>(compiler.get_declared_struct_size_runtime_array(spirv_type, 0));
}

inline void read_resource_size(const spirv_cross::Compiler &compiler, const spirv_cross::Resource &resource, Shader::Constant &descriptor)
{
	const auto &spirv_type = compiler.get_type_from_variable(resource.id);

	descriptor.size = static_cast<uint32_t>(compiler.get_declared_struct_size_runtime_array(spirv_type, 0));
}

inline void read_resource_size(const spirv_cross::Compiler &compiler, const spirv_cross::SPIRConstant &constant, Shader::Constant &descriptor)
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
inline std::vector<T> read_shader_resource(const spirv_cross::Compiler &compiler, VkShaderStageFlags stage)
{
	LOG_ERROR("Not implemented! Read shader resources of type.");
}

template <>
inline std::vector<Shader::Attribute> read_shader_resource<Shader::Attribute>(const spirv_cross::Compiler &compiler, VkShaderStageFlags stage)
{
	std::vector<Shader::Attribute> attributes;

	// Parsing input attribute
	auto input_resources = compiler.get_shader_resources().stage_inputs;
	for (auto &resource : input_resources)
	{
		Shader::Attribute attribute{};
		attribute.type  = Shader::Attribute::Type::Input;
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
		Shader::Attribute attribute{};
		attribute.type  = Shader::Attribute::Type::Output;
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
inline std::vector<Shader::InputAttachment> read_shader_resource<Shader::InputAttachment>(const spirv_cross::Compiler &compiler, VkShaderStageFlags stage)
{
	std::vector<Shader::InputAttachment> input_attachments;

	auto subpass_resources = compiler.get_shader_resources().subpass_inputs;
	for (auto &resource : subpass_resources)
	{
		Shader::InputAttachment input_attachment{};
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
inline std::vector<Shader::Image> read_shader_resource<Shader::Image>(const spirv_cross::Compiler &compiler, VkShaderStageFlags stage)
{
	std::vector<Shader::Image> images;

	// Parsing image
	auto image_resources = compiler.get_shader_resources().separate_images;
	for (auto &resource : image_resources)
	{
		Shader::Image image{};
		image.type  = Shader::Image::Type::Image;
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
		Shader::Image image{};
		image.type  = Shader::Image::Type::ImageSampler;
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
		Shader::Image image{};
		image.type  = Shader::Image::Type::ImageStorage;
		image.stage = stage;
		image.name  = resource.name;

		read_resource_array_size(compiler, resource, image);
		read_resource_decoration<spv::DecorationNonReadable>(compiler, resource, image);
		read_resource_decoration<spv::DecorationNonWritable>(compiler, resource, image);
		read_resource_decoration<spv::DecorationDescriptorSet>(compiler, resource, image);
		read_resource_decoration<spv::DecorationBinding>(compiler, resource, image);

		images.push_back(image);
	}

	// Parsing sampler
	image_resources = compiler.get_shader_resources().separate_samplers;
	for (auto &resource : image_resources)
	{
		Shader::Image image{};
		image.type  = Shader::Image::Type::Sampler;
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
inline std::vector<Shader::Buffer> read_shader_resource<Shader::Buffer>(const spirv_cross::Compiler &compiler, VkShaderStageFlags stage)
{
	std::vector<Shader::Buffer> buffers;

	// Parsing uniform buffer
	auto uniform_resources = compiler.get_shader_resources().uniform_buffers;
	for (auto &resource : uniform_resources)
	{
		Shader::Buffer buffer{};
		buffer.type  = Shader::Buffer::Type::Uniform;
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
		Shader::Buffer buffer{};
		buffer.type  = Shader::Buffer::Type::Storage;
		buffer.stage = stage;
		buffer.name  = resource.name;

		read_resource_size(compiler, resource, buffer);
		read_resource_array_size(compiler, resource, buffer);
		read_resource_decoration<spv::DecorationNonReadable>(compiler, resource, buffer);
		read_resource_decoration<spv::DecorationNonWritable>(compiler, resource, buffer);
		read_resource_decoration<spv::DecorationDescriptorSet>(compiler, resource, buffer);
		read_resource_decoration<spv::DecorationBinding>(compiler, resource, buffer);

		buffers.push_back(buffer);
	}

	return buffers;
}

template <>
inline std::vector<Shader::Constant> read_shader_resource<Shader::Constant>(const spirv_cross::Compiler &compiler, VkShaderStageFlags stage)
{
	std::vector<Shader::Constant> constants;

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

		Shader::Constant constant{};
		constant.type   = Shader::Constant::Type::Push;
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

		Shader::Constant constant{};
		constant.type        = Shader::Constant::Type::Specialization;
		constant.stage       = stage;
		constant.name        = compiler.get_name(resource.id);
		constant.offset      = 0;
		constant.constant_id = resource.constant_id;

		read_resource_size(compiler, spirv_value, constant);

		constants.push_back(constant);
	}

	return constants;
}

Shader::~Shader()
{
	for (auto &shader_module : m_shader_module_cache)
	{
		if (shader_module != VK_NULL_HANDLE)
		{
			vkDestroyShaderModule(Engine::instance()->getContext().getSubsystem<GraphicsContext>()->getLogicalDevice(), shader_module, nullptr);
		}
	}

	m_shader_module_cache.clear();
}

VkShaderModule Shader::createShaderModule(const std::string &filename, const Variant &variant)
{
	auto stage = getShaderStage(filename);
	auto type  = getShaderFileType(filename);

	if (m_stage & stage)
	{
		VK_INFO("Shader module already exist");
		return m_shader_module_cache[stage];
	}

	m_stage |= stage;

	std::vector<uint8_t> raw_data;
	FileSystem::read(filename, raw_data, type == ShaderFileType::SPIRV);

	std::vector<uint32_t> spirv;
	switch (type)
	{
		case ShaderFileType::GLSL:
			spirv = compileGLSL(raw_data, stage, variant);
			break;
		case ShaderFileType::HLSL:
			spirv = compileHLSL(raw_data, stage, variant);
			break;
		case ShaderFileType::SPIRV:
			spirv.resize(raw_data.size() / 4);
			std::memcpy(spirv.data(), raw_data.data(), raw_data.size());
			break;
		default:
			break;
	}

	reflectSpirv(spirv, stage);

	VkShaderModuleCreateInfo shader_module_create_info = {};
	shader_module_create_info.sType                    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
	shader_module_create_info.codeSize                 = spirv.size() * sizeof(uint32_t);
	shader_module_create_info.pCode                    = spirv.data();

	VkShaderModule shader_module;
	if (!VK_CHECK(vkCreateShaderModule(Engine::instance()->getContext().getSubsystem<GraphicsContext>()->getLogicalDevice(), &shader_module_create_info, nullptr, &shader_module)))
	{
		VK_ERROR("Failed to create shader module");
		return VK_NULL_HANDLE;
	}

	m_shader_module_cache.push_back(shader_module);

	// Update hash
	for (auto &image : m_images)
	{
		hash_combine(m_hash, image.hash());
	}

	for (auto &buffer : m_buffers)
	{
		hash_combine(m_hash, buffer.hash());
	}

	for (auto &input_attachment : m_input_attachments)
	{
		hash_combine(m_hash, input_attachment.hash());
	}

	return shader_module;
}

size_t Shader::hash() const
{
	return m_hash;
}

std::vector<VkPushConstantRange> Shader::getPushConstantRanges() const
{
	std::vector<VkPushConstantRange> push_constant_ranges;
	for (auto &constant : m_constants)
	{
		if (constant.type == Constant::Type::Push)
		{
			VkPushConstantRange push_constant_range = {};
			push_constant_range.stageFlags          = constant.stage;
			push_constant_range.size                = constant.size;
			push_constant_range.offset              = constant.offset;
			push_constant_ranges.push_back(push_constant_range);
		}
	}

	return push_constant_ranges;
}

const std::unordered_set<uint32_t> &Shader::getSets() const
{
	return m_sets;
}

const std::unordered_map<VkShaderStageFlags, std::vector<Shader::Attribute>> &Shader::getAttributeReflection() const
{
	return m_attributes;
}

const std::vector<Shader::Image> &Shader::getImageReflection() const
{
	return m_images;
}

const std::vector<Shader::Buffer> &Shader::getBufferReflection() const
{
	return m_buffers;
}

const std::vector<Shader::Constant> &Shader::getConstantReflection() const
{
	return m_constants;
}

const std::vector<Shader::InputAttachment> &Shader::getInputAttachmentReflection() const
{
	return m_input_attachments;
}

void Shader::setBufferMode(uint32_t set, uint32_t binding, ShaderResourceMode mode)
{
	for (auto &buffer : m_buffers)
	{
		if (buffer.binding == binding && buffer.set == set)
		{
			buffer.mode = mode;
			return;
		}
	}
}

VkShaderStageFlagBits Shader::getShaderStage(const std::string &filename)
{
	auto extension = FileSystem::getFileExtension(filename);

	if (extension == ".vert")
	{
		return VK_SHADER_STAGE_VERTEX_BIT;
	}
	else if (extension == ".frag")
	{
		return VK_SHADER_STAGE_FRAGMENT_BIT;
	}
	else if (extension == ".tesc")
	{
		return VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT;
	}
	else if (extension == ".tese")
	{
		return VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT;
	}
	else if (extension == ".geom")
	{
		return VK_SHADER_STAGE_GEOMETRY_BIT;
	}
	else if (extension == ".comp")
	{
		return VK_SHADER_STAGE_COMPUTE_BIT;
	}
	else if (extension == ".rgen")
	{
		return VK_SHADER_STAGE_RAYGEN_BIT_KHR;
	}
	else if (extension == ".rahit")
	{
		return VK_SHADER_STAGE_ANY_HIT_BIT_KHR;
	}
	else if (extension == ".rchit")
	{
		return VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
	}
	else if (extension == ".rmiss")
	{
		return VK_SHADER_STAGE_MISS_BIT_KHR;
	}
	else if (extension == ".rint")
	{
		return VK_SHADER_STAGE_INTERSECTION_BIT_KHR;
	}
	else if (extension == ".rcall")
	{
		return VK_SHADER_STAGE_CALLABLE_BIT_KHR;
	}
	else if (extension == ".mesh")
	{
		return VK_SHADER_STAGE_MESH_BIT_NV;
	}
	else if (extension == ".task")
	{
		return VK_SHADER_STAGE_TASK_BIT_NV;
	}

	return VK_SHADER_STAGE_ALL;
}

Shader::ShaderFileType Shader::getShaderFileType(const std::string &filename)
{
	auto extension = FileSystem::getFileExtension(FileSystem::getFileName(filename));

	if (extension == ".glsl")
	{
		return ShaderFileType::GLSL;
	}
	else if (extension == ".hlsl")
	{
		return ShaderFileType::HLSL;
	}
	else if (extension == ".spv")
	{
		return ShaderFileType::SPIRV;
	}

	return ShaderFileType::GLSL;
}

std::vector<uint32_t> Shader::compileGLSL(const std::vector<uint8_t> &data, VkShaderStageFlags stage, const Variant &variant)
{
	glslang::InitializeProcess();

	EShMessages msgs = static_cast<EShMessages>(EShMsgDefault | EShMsgVulkanRules | EShMsgSpvRules);

	EShLanguage lang = get_shader_language(stage);

	std::string info_log = "";

	const char *file_name_list[1] = {""};
	const char *shader_source     = reinterpret_cast<const char *>(data.data());

	glslang::TShader shader(lang);
	shader.setStringsWithLengthsAndNames(&shader_source, nullptr, file_name_list, 1);
	shader.setEntryPoint("main");
	shader.setSourceEntryPoint("main");
	shader.setEnvTarget(glslang::EShTargetSpv, glslang::EShTargetSpv_1_4);
	shader.setPreamble(variant.getPreamble().c_str());
	shader.addProcesses(variant.getProcesses());

	if (!shader.parse(&glslang::DefaultTBuiltInResource, 100, false, msgs))
	{
		info_log = std::string(shader.getInfoLog()) + "\n" + std::string(shader.getInfoDebugLog());
		LOG_ERROR(info_log);
		return {};
	}

	// Add shader to a new program
	glslang::TProgram program;
	program.addShader(&shader);

	// Link program
	if (!program.link(msgs))
	{
		info_log = std::string(program.getInfoLog()) + "\n" + std::string(program.getInfoDebugLog());
		LOG_ERROR(info_log);
		return {};
	}

	// Save info log
	if (shader.getInfoLog())
	{
		info_log = std::string(shader.getInfoLog()) + "\n" + std::string(shader.getInfoDebugLog());
	}

	if (program.getInfoLog())
	{
		info_log = std::string(program.getInfoLog()) + "\n" + std::string(program.getInfoDebugLog());
	}

	glslang::TIntermediate *intermediate = program.getIntermediate(lang);

	if (!intermediate)
	{
		info_log += "Failed to get shared intermediate code!\n";
		LOG_ERROR(info_log);
		return {};
	}

	spv::SpvBuildLogger logger;

	std::vector<uint32_t> spirv;
	std::vector<uint8_t>  result;

	glslang::GlslangToSpv(*intermediate, spirv, &logger);
	result.resize(spirv.size() * 4);
	std::memcpy(result.data(), spirv.data(), result.size());

	info_log += logger.getAllMessages() + "\n";

	glslang::FinalizeProcess();
	return spirv;
}

std::vector<uint32_t> Shader::compileHLSL(const std::vector<uint8_t> &data, VkShaderStageFlags stage, const Variant &variant)
{
	// TODO:
	return {};
}

void Shader::reflectSpirv(const std::vector<uint32_t> &spirv, VkShaderStageFlags stage)
{
	spirv_cross::CompilerGLSL compiler(spirv);

	auto opts                     = compiler.get_common_options();
	opts.enable_420pack_extension = true;

	compiler.set_common_options(opts);

	auto attributes        = read_shader_resource<Shader::Attribute>(compiler, stage);
	auto input_attachments = read_shader_resource<Shader::InputAttachment>(compiler, stage);
	auto images            = read_shader_resource<Shader::Image>(compiler, stage);
	auto buffers           = read_shader_resource<Shader::Buffer>(compiler, stage);
	auto constants         = read_shader_resource<Shader::Constant>(compiler, stage);

	m_attributes[stage] = attributes;

#define ADD_RESOURCE(resources, m_resources)  \
	for (auto &resource : resources)          \
	{                                         \
		bool exist = false;                   \
		for (auto &item : m_resources)        \
		{                                     \
			if (resource == item)             \
			{                                 \
				item.stage |= resource.stage; \
				exist = true;                 \
				break;                        \
			}                                 \
		}                                     \
		if (!exist)                           \
		{                                     \
			m_resources.push_back(resource);  \
		}                                     \
	}

	ADD_RESOURCE(input_attachments, m_input_attachments);
	ADD_RESOURCE(images, m_images);
	ADD_RESOURCE(buffers, m_buffers);
	ADD_RESOURCE(constants, m_constants);

	// Check binding and set conflict
	std::for_each(m_images.begin(), m_images.end(), [this](const Shader::Image &image) { m_sets.insert(image.set); });
	std::for_each(m_buffers.begin(), m_buffers.end(), [this](const Shader::Buffer &buffer) { m_sets.insert(buffer.set); });

	for (auto &image : m_images)
	{
		for (auto &buffer : m_buffers)
		{
			if (image.set == buffer.set && image.binding == buffer.binding)
			{
				VK_WARN("Set {}, Bind {} has image and buffer in the same time!");
				return;
			}
		}
	}
}

size_t Shader::Variant::hash() const
{
	std::hash<std::string> hasher{};
	return hasher(m_preamble);
}

void Shader::Variant::addDefinitions(const std::vector<std::string> &definitions)
{
	for (auto &definition : definitions)
	{
		addDefine(definition);
	}
}

void Shader::Variant::addDefine(const std::string &def)
{
	m_processes.push_back("D" + def);

	std::string tmp_def = def;

	size_t pos_equal = tmp_def.find_first_of("=");
	if (pos_equal != std::string::npos)
	{
		tmp_def[pos_equal] = ' ';
	}

	m_preamble.append("#define " + tmp_def + "\n");
}

void Shader::Variant::addUndefine(const std::string &undef)
{
	m_processes.push_back("U" + undef);
	m_preamble.append("#undef " + undef + "\n");
}

const std::string &Shader::Variant::getPreamble() const
{
	return m_preamble;
}

const std::vector<std::string> &Shader::Variant::getProcesses() const
{
	return m_processes;
}

void Shader::Variant::clear()
{
	m_preamble.clear();
	m_processes.clear();
}
}        // namespace Ilum