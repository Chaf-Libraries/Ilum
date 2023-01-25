#pragma once

#include "Precompile.hpp"

namespace Ilum
{
class  Variant
{
  public:
	Variant() = default;

	~Variant();

	Variant(Variant &&other) noexcept;

	Variant(const Variant &other);

	template <typename _Ty>
	Variant(const _Ty& var)
	{
		Set(&var, sizeof(_Ty));
	}

	Variant &operator=(Variant &&other) noexcept;

	Variant &operator=(const Variant &other);

	bool Empty() const;

	template <typename _Ty>
	void operator=(const _Ty &var)
	{
		Set(&var, sizeof(_Ty));
	}

	template <typename _Ty>
	_Ty *Convert() const
	{
		return std::static_pointer_cast<_Ty>(m_data).get();
	}

	template <class Archive>
	void save(Archive &archive) const
	{
		std::vector<uint8_t> data(m_size);
		std::memcpy(data.data(), m_data.get(), m_size);
		archive(data);
	}

	template <class Archive>
	void load(Archive &archive)
	{
		std::vector<uint8_t> data;
		archive(data);
		Set(data.data(), data.size());
	}

  private:
	void Set(const void *data, size_t size);

  private:
	std::shared_ptr<void> m_data = nullptr;

	size_t m_size = 0;
};
}        // namespace Ilum