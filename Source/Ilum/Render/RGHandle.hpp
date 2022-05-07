#pragma once

#include <RHI/Texture.hpp>
#include <RHI/Buffer.hpp>

namespace Ilum
{
class RGHandle
{
  public:
	explicit RGHandle();
	RGHandle(uint32_t index);
	~RGHandle() = default;

	operator uint32_t() const;

	void Invalidate();
	bool IsInvalid() const;

	bool operator<(const RGHandle &rhs);

	bool operator==(const RGHandle &rhs);

	static void Reset();

	static void SetCurrent(uint32_t id);

  private:
	inline static uint32_t INVALID_ID = ~0U;
	inline static uint32_t CURRENT_ID = 0U;
	uint32_t               m_index    = INVALID_ID;
};

enum class ResourceType
{
	None,
	Texture,
	Buffer
};

class ResourceDeclaration
{
  public:
	explicit ResourceDeclaration(const std::string &name, ResourceType type = ResourceType::None) :
	    m_name(name), m_type(type)
	{
	}

	inline const std::string &GetName() const
	{
		return m_name;
	}

	inline ResourceType GetType() const
	{
		return m_type;
	}

  private:
	std::string  m_name;
	ResourceType m_type;
};

class TextureDeclaration : public ResourceDeclaration
{
  public:
	TextureDeclaration(const std::string &name, const TextureDesc &desc, const TextureState &state) :
	    ResourceDeclaration(name, ResourceType::Texture), m_desc(desc), m_state(state)
	{
	}

	inline const TextureDesc &GetDesc() const
	{
		return m_desc;
	}

	inline const TextureState &GetState() const
	{
		return m_state;
	}

  private:
	TextureDesc  m_desc;
	TextureState m_state;
};

class BufferDeclaration : public ResourceDeclaration
{
  public:
	BufferDeclaration(const std::string &name, const BufferDesc &desc, const BufferState &state) :
	    ResourceDeclaration(name, ResourceType::Buffer), m_desc(desc), m_state(state)
	{
	}

	inline const BufferDesc &GetDesc() const
	{
		return m_desc;
	}

	inline const BufferState &GetState() const
	{
		return m_state;
	}

  private:
	BufferDesc  m_desc;
	BufferState m_state;
};
}