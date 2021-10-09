#pragma once

#include "Core/Engine/PCH.hpp"

#include <glslang/Include/ResourceLimits.h>
#include <glslang/SPIRV/GLSL.std.450.h>
#include <glslang/SPIRV/GlslangToSpv.h>

#include <spirv_glsl.hpp>

namespace Ilum
{
class LogicalDevice;

class Shader
{
  public:
	enum class ShaderFileType
	{
		GLSL,
		HLSL,
		SPIRV
	};

	enum class ShaderCompileState
	{
		Idle,
		Compiling,
		Success,
		Failed
	};

	enum class ShaderResourceMode
	{
		Static,
		Dynamic,
		UpdateAfterBind
	};

	struct ShaderResourceQualifiers
	{
		enum : uint32_t
		{
			None        = 0,
			NonReadable = 1,
			NonWritable = 2
		};
	};

	class Variant
	{
	  public:
		Variant() = default;

		~Variant() = default;

		size_t hash() const;

		void addDefinitions(const std::vector<std::string> &definitions);

		void addDefine(const std::string &def);

		void addUndefine(const std::string &undef);

		const std::string &getPreamble() const;

		const std::vector<std::string> &getProcesses() const;

		void clear();

	  private:
		size_t                   m_hash = 0;
		std::string              m_preamble;
		std::vector<std::string> m_processes;
	};

	// TODO: Remove in the future
	struct ShaderDescription
	{
		std::unordered_map<uint32_t, std::vector<VkDescriptorSetLayoutBinding>> m_descriptor_set_layout_bindings;
		std::vector<VkDescriptorPoolSize>                                       m_descriptor_pool_sizes;
		std::vector<VkVertexInputBindingDescription>                            m_vertex_input_binding_descriptions;
		std::vector<VkVertexInputAttributeDescription>                          m_vertex_input_attribute_descriptions;

		void clear();
	};

	struct Attribute
	{
		enum class Type
		{
			None,
			Input,
			Output
		};

		std::string        name       = "";
		uint32_t           location   = 0;
		uint32_t           vec_size   = 0;
		uint32_t           array_size = 0;
		uint32_t           columns    = 0;
		Type               type       = Type::None;
		VkShaderStageFlags stage      = VK_SHADER_STAGE_ALL;
		VkVertexInputRate  rate       = VK_VERTEX_INPUT_RATE_VERTEX;
		uint32_t           base_type  = 0;

		inline bool operator==(const Attribute &rhs)
		{
			return type == rhs.type &&
			       location == rhs.location &&
			       vec_size == rhs.vec_size &&
			       array_size == rhs.array_size &&
			       columns == rhs.columns;
		}
	};

	struct InputAttachment
	{
		std::string        name                   = "";
		uint32_t           array_size             = 0;
		uint32_t           input_attachment_index = 0;
		uint32_t           set                    = 0;
		uint32_t           binding                = 0;
		VkShaderStageFlags stage                  = VK_SHADER_STAGE_FRAGMENT_BIT;

		inline bool operator==(const InputAttachment &rhs)
		{
			return array_size == rhs.array_size &&
			       input_attachment_index == rhs.input_attachment_index &&
			       binding == rhs.binding;
		}

		inline size_t hash()
		{
			size_t seed = 0;
			hash_combine(seed, name);
			hash_combine(seed, array_size);
			hash_combine(seed, input_attachment_index);
			hash_combine(seed, set);
			hash_combine(seed, binding);
			hash_combine(seed, stage);

			return seed;
		}
	};

	struct Image
	{
		enum class Type
		{
			None,
			ImageSampler,
			Image,
			ImageStorage,
			Sampler
		};

		std::string        name       = "";
		uint32_t           array_size = 0;        // When array_size = 0, enable descriptor indexing
		uint32_t           set        = 0;
		uint32_t           binding    = 0;
		VkShaderStageFlags stage      = VK_SHADER_STAGE_ALL;
		Type               type       = Type::None;

		// Only for storage image
		uint32_t qualifiers = 0;

		inline bool operator==(const Image &rhs)
		{
			return type == rhs.type &&
			       array_size == rhs.array_size &&
			       set == rhs.set &&
			       binding == rhs.binding &&
			       (type == Type::ImageStorage ? qualifiers == rhs.qualifiers : true);
		}

		inline size_t hash()
		{
			size_t seed = 0;
			hash_combine(seed, name);
			hash_combine(seed, array_size);
			hash_combine(seed, set);
			hash_combine(seed, binding);
			hash_combine(seed, stage);
			hash_combine(seed, type);
			hash_combine(seed, qualifiers);

			return seed;
		}
	};

	struct Buffer
	{
		enum class Type
		{
			None,
			Uniform,
			Storage
		};

		std::string        name       = "";
		uint32_t           size       = 0;
		uint32_t           array_size = 0;        // When array_size = 0, enable descriptor indexing
		uint32_t           set        = 0;
		uint32_t           binding    = 0;
		VkShaderStageFlags stage      = VK_SHADER_STAGE_ALL;
		Type               type       = Type::None;
		ShaderResourceMode mode       = ShaderResourceMode::Static;

		// Qualifier only avaliable for storage image
		uint32_t qualifiers = 0;

		inline bool operator==(const Buffer &rhs)
		{
			return type == rhs.type &&
			       size == rhs.size &&
			       array_size == rhs.array_size &&
			       set == rhs.set &&
			       binding == rhs.binding &&
			       (type == Type::Storage ? qualifiers == rhs.qualifiers : true);
		}

		inline size_t hash()
		{
			size_t seed = 0;
			hash_combine(seed, name);
			hash_combine(seed, size);
			hash_combine(seed, array_size);
			hash_combine(seed, set);
			hash_combine(seed, binding);
			hash_combine(seed, stage);
			hash_combine(seed, type);
			hash_combine(seed, mode);
			hash_combine(seed, qualifiers);

			return seed;
		}
	};

	struct Constant
	{
		enum class Type
		{
			None,
			Push,
			Specialization
		};

		std::string        name   = "";
		uint32_t           size   = 0;
		uint32_t           offset = 0;
		VkShaderStageFlags stage  = VK_SHADER_STAGE_ALL;
		Type               type   = Type::None;

		// Only for Specialization constant
		uint32_t constant_id = 0;

		inline bool operator==(const Constant &rhs)
		{
			return type == rhs.type &&
			       size == rhs.size &&
			       offset == rhs.offset &&
			       (type == Type::Specialization ? constant_id == rhs.constant_id : true);
		}
	};

  private:
	inline static const std::unordered_map<uint32_t, uint32_t> base_type_size = {
	    {spirv_cross::SPIRType::BaseType::SByte, 1},
	    {spirv_cross::SPIRType::BaseType::UByte, 1},
	    {spirv_cross::SPIRType::BaseType::Short, 2},
	    {spirv_cross::SPIRType::BaseType::UShort, 2},
	    {spirv_cross::SPIRType::BaseType::Int, 4},
	    {spirv_cross::SPIRType::BaseType::UInt, 4},
	    {spirv_cross::SPIRType::BaseType::Int64, 8},
	    {spirv_cross::SPIRType::BaseType::UInt64, 8},
	    {spirv_cross::SPIRType::BaseType::Float, 4},
	    {spirv_cross::SPIRType::BaseType::Double, 8}};

	inline static const std::unordered_map<uint32_t, std::vector<VkFormat>> attribute_format = {
	    {spirv_cross::SPIRType::BaseType::SByte, {VK_FORMAT_UNDEFINED, VK_FORMAT_R8_SINT, VK_FORMAT_R8G8_SINT, VK_FORMAT_R8G8B8_SINT, VK_FORMAT_R8G8B8A8_SINT}},
	    {spirv_cross::SPIRType::BaseType::UByte, {VK_FORMAT_UNDEFINED, VK_FORMAT_R8_UINT, VK_FORMAT_R8G8_UINT, VK_FORMAT_R8G8B8_UINT, VK_FORMAT_R8G8B8A8_UINT}},
	    {spirv_cross::SPIRType::BaseType::Short, {VK_FORMAT_UNDEFINED, VK_FORMAT_R16_SINT, VK_FORMAT_R16G16_SINT, VK_FORMAT_R16G16B16_SINT, VK_FORMAT_R16G16B16A16_SINT}},
	    {spirv_cross::SPIRType::BaseType::UShort, {VK_FORMAT_UNDEFINED, VK_FORMAT_R16_UINT, VK_FORMAT_R16G16_UINT, VK_FORMAT_R16G16B16_UINT, VK_FORMAT_R16G16B16A16_UINT}},
	    {spirv_cross::SPIRType::BaseType::Int, {VK_FORMAT_UNDEFINED, VK_FORMAT_R32_SINT, VK_FORMAT_R32G32_SINT, VK_FORMAT_R32G32B32_SINT, VK_FORMAT_R32G32B32A32_SINT}},
	    {spirv_cross::SPIRType::BaseType::UInt, {VK_FORMAT_UNDEFINED, VK_FORMAT_R32_UINT, VK_FORMAT_R32G32_UINT, VK_FORMAT_R32G32B32_UINT, VK_FORMAT_R32G32B32A32_UINT}},
	    {spirv_cross::SPIRType::BaseType::Int64, {VK_FORMAT_UNDEFINED, VK_FORMAT_R64_SINT, VK_FORMAT_R64G64_SINT, VK_FORMAT_R64G64B64_SINT, VK_FORMAT_R64G64B64A64_SINT}},
	    {spirv_cross::SPIRType::BaseType::UInt64, {VK_FORMAT_UNDEFINED, VK_FORMAT_R64_UINT, VK_FORMAT_R64G64_UINT, VK_FORMAT_R64G64B64_UINT, VK_FORMAT_R64G64B64A64_UINT}},
	    {spirv_cross::SPIRType::BaseType::Float, {VK_FORMAT_UNDEFINED, VK_FORMAT_R32_SFLOAT, VK_FORMAT_R32G32_SFLOAT, VK_FORMAT_R32G32B32_SFLOAT, VK_FORMAT_R32G32B32A32_SFLOAT}},
	    {spirv_cross::SPIRType::BaseType::Double, {VK_FORMAT_UNDEFINED, VK_FORMAT_R64_SFLOAT, VK_FORMAT_R64G64_SFLOAT, VK_FORMAT_R64G64B64_SFLOAT, VK_FORMAT_R64G64B64A64_SFLOAT}}};

  public:
	Shader() = default;

	~Shader();

	VkShaderModule createShaderModule(const std::string &filename, const Variant &variant = {});

	const ShaderDescription &getShaderDescription() const;

	size_t hash() const;

	std::vector<VkPushConstantRange> getPushConstantRanges() const;

  public:
	const std::unordered_set<uint32_t> &getSets() const;

	const std::unordered_map<VkShaderStageFlags, std::vector<Attribute>> &getAttributeReflection() const;

	const std::vector<Image> &getImageReflection() const;

	const std::vector<Buffer> &getBufferReflection() const;

	const std::vector<Constant> &getConstantReflection() const;

	const std::vector<InputAttachment> &getInputAttachmentReflection() const;

	void setBufferMode(uint32_t set, uint32_t binding, ShaderResourceMode mode);

	// Vertex input state:
	// binding#0 - VK_VERTEX_INPUT_RATE_VERTEX
	// binding#1 - VK_VERTEX_INPUT_RATE_INSTANCE
	template <typename Vertex, typename Instance = void>
	void setVertexInput()
	{
		m_vertex_stride   = (typeid(Vertex) == typeid(void) ? 0 : sizeof(Vertex));
		m_instance_stride = (typeid(Instance) == typeid(void) ? 0 : sizeof(Instance));
	}

  public:
	// Shader file naming: shader_name.shader_file_type.shader_stage
	static VkShaderStageFlagBits getShaderStage(const std::string &filename);

	static ShaderFileType getShaderFileType(const std::string &filename);

	static std::vector<uint32_t> compileGLSL(const std::vector<uint8_t> &data, VkShaderStageFlags stage, const Variant &variant = {});

	static std::vector<uint32_t> compileHLSL(const std::vector<uint8_t> &data, VkShaderStageFlags stage, const Variant &variant = {});

  private:
	void reflectSpirv(const std::vector<uint32_t> &spirv, VkShaderStageFlags stage);

	void updateShaderDescription();

  private:
	ShaderFileType m_shader_type = ShaderFileType::GLSL;

	VkShaderStageFlags m_stage = 0;

	uint32_t m_vertex_stride = 0;

	uint32_t m_instance_stride = 0;

	// Shader resources descriptions
	std::unordered_map<VkShaderStageFlags, std::vector<Attribute>> m_attributes;
	std::vector<InputAttachment>                                   m_input_attachments;
	std::vector<Image>                                             m_images;
	std::vector<Buffer>                                            m_buffers;
	std::vector<Constant>                                          m_constants;

	// Descriptor set index
	std::unordered_set<uint32_t> m_sets;

	ShaderCompileState          m_compile_state = ShaderCompileState::Idle;
	std::vector<VkShaderModule> m_shader_module_cache;

	ShaderDescription m_shader_description;

	size_t m_hash = 0;
};
}        // namespace Ilum