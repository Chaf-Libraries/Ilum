#pragma once

#include "Precompile.hpp"

namespace Ilum
{
class EXPORT_API Variant
{
  public:
	Variant() = default;

	~Variant();

	Variant(Variant &&other) noexcept;

	Variant(const Variant &other);

	Variant &operator=(Variant &&other) noexcept;

	Variant &operator=(const Variant &other);

	template <typename _Ty>
	void operator=(const _Ty &var)
	{
		Set(&var, sizeof(_Ty));
	}

	template <typename _Ty>
	_Ty &Convert()
	{
		return *static_cast<_Ty *>(m_data);
	}

	template <class Archive>
	void save(Archive &archive) const
	{
		std::vector<uint8_t> data(m_size);
		std::memcpy(data.data(), m_data, m_size);
		archive(data);
	}

	template <class Archive>
	void load(Archive &archive)
	{
		std::vector<uint8_t> data;
		archive(data);
		std::memcpy(m_data, data.data(), m_size);
	}

  private:
	void Set(const void *data, size_t size);

  private:
	void           *m_data = nullptr;
	size_t          m_size = 0;
};
}        // namespace Ilum