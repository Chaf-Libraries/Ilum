#pragma once

#include "Core/Engine/PCH.hpp"

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

		size_t getID() const;

		void addDefinitions(const std::vector<std::string> &definitions);

		void addDefine(const std::string &def);

		void addUndefine(const std::string &undef);

		const std::string &getPreamble() const;

		const std::vector<std::string> &getProcesses() const;

		void clear();

	  private:
		void updateID();

	  private:
		size_t                   m_id = 0;
		std::string              m_preamble;
		std::vector<std::string> m_processes;
	};

	struct ShaderDescription
	{
		std::unordered_map<uint32_t, std::vector<VkDescriptorSetLayoutBinding>> m_descriptor_set_layout_bindings;
		std::vector<VkDescriptorPoolSize>                                       m_descriptor_pool_sizes;
		std::vector<VkVertexInputBindingDescription>                            m_vertex_input_binding_descriptions;
		std::vector<VkVertexInputAttributeDescription>                          m_vertex_input_attribute_descriptions;
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
		uint32_t           binding                = 0;
		VkShaderStageFlags stage                  = VK_SHADER_STAGE_FRAGMENT_BIT;

		inline bool operator==(const InputAttachment &rhs)
		{
			return array_size == rhs.array_size &&
			       input_attachment_index == rhs.input_attachment_index &&
			       binding == rhs.binding;
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

  public:
	Shader(const LogicalDevice &logical_device);

	~Shader();

	VkShaderModule createShaderModule(const std::string &filename, const Variant &variant = {});

	ShaderDescription createShaderDescription();

  public:
	const std::unordered_map<VkShaderStageFlags, std::vector<Attribute>> &getAttributeReflection() const;

	const std::vector<Image> &getImageReflection() const;

	const std::vector<Buffer> &getBufferReflection() const;

	const std::vector<Constant> &getConstantReflection() const;

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
	static VkShaderStageFlags getShaderStage(const std::string &filename);

	static ShaderFileType getShaderFileType(const std::string &filename);

	static std::vector<uint32_t> compileGLSL(const std::vector<uint8_t> &data, VkShaderStageFlags stage, const Variant &variant = {});

	static std::vector<uint32_t> compileHLSL(const std::vector<uint8_t> &data, VkShaderStageFlags stage, const Variant &variant = {});

  private:
	void reflectSpirv(const std::vector<uint32_t> &spirv, VkShaderStageFlags stage);

  private:
	const LogicalDevice &m_logical_device;

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

	ShaderCompileState          m_compile_state = ShaderCompileState::Idle;
	std::vector<VkShaderModule> m_shader_module_cache;
};
}        // namespace Ilum