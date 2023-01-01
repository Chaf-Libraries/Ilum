#include "Variant.hpp"

#include <any>

namespace Ilum
{
Variant::~Variant()
{
	if (m_data)
	{
		m_data.reset();
		m_size = 0;
	}
}

Variant::Variant(Variant &&other) noexcept :
    m_data(std::move(other.m_data)),
    m_size(other.m_size)
{
	other.m_data = nullptr;
}

Variant::Variant(const Variant &other) :
    m_size(other.m_size), m_data(other.m_data)
{
}

Variant &Variant::operator=(Variant &&other) noexcept
{
	m_data = std::move(other.m_data);
	m_size = other.m_size;

	other.m_data = nullptr;

	return *this;
}

Variant &Variant::operator=(const Variant &other)
{
	m_data = other.m_data;
	m_size = other.m_size;

	return *this;
}

bool Variant::Empty() const
{
	return m_data == nullptr;
}

void Variant::Set(const void *data, size_t size)
{
	if (m_size < size)
	{
		m_data = std::shared_ptr<void>(malloc(size));
		m_size = size;
	}

	if (m_data)
	{
		std::memcpy(m_data.get(), data, size);
	}
}
}        // namespace Ilum