#include "Variant.hpp"

#include <any>

namespace Ilum
{
Variant::~Variant()
{
	if (m_data)
	{
		delete m_data;
		m_data = nullptr;
	}
}

Variant::Variant(Variant &&other) noexcept :
    m_data(other.m_data),
    m_size(other.m_size)
{
	other.m_data = nullptr;
}

Variant::Variant(const Variant &other) :
    m_size(other.m_size)
{
	m_data = malloc(m_size);
	if (m_data)
	{
		std::memcpy(m_data, other.m_data, m_size);
	}
}

Variant &Variant::operator=(Variant &&other) noexcept
{
	m_data = other.m_data;
	m_size = other.m_size;

	other.m_data = nullptr;

	return *this;
}

Variant &Variant::operator=(const Variant &other)
{
	m_data = other.m_data;
	m_size = other.m_size;

	if (m_size < other.m_size)
	{
		m_data = realloc(m_data, other.m_size);
		m_size = other.m_size;
	}

	if (m_data)
	{
		std::memcpy(m_data, other.m_data, other.m_size);
	}

	return *this;
}

void Variant::Set(const void *data, size_t size)
{
	if (m_size < size)
	{
		void *new_data = realloc(m_data, size);
		if (new_data)
		{
			m_data = new_data;
		}
		m_size = size;
	}

	if (m_data)
	{
		std::memcpy(m_data, data, size);
	}
}
}        // namespace Ilum