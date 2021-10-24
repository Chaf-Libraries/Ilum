#pragma once

#include <entt.hpp>

#include "Entity.hpp"

namespace Ilum
{
template <typename T>
class EntityView
{
	class iterator;

  public:
	EntityView() :
	    m_view(Scene::instance()->getRegistry().view<T>())
	{
	}

	Entity operator[](size_t index)
	{
		ASSERT(index < size() && "Index out of range of Entity View");
		return Entity(m_view[index]);
	}

	bool empty() const
	{
		return m_view.empty();
	}

	size_t size() const
	{
		return m_view.size();
	}

	Entity front()
	{
		Entity(m_view[0]);
	}

	iterator begin()
	{
		return EntityView<T>::iterator(*this, 0);
	}

	iterator end()
	{
		return EntityView<T>::iterator(*this, size());
	}

	const iterator begin() const
	{
		return EntityView<T>::iterator(*this, 0);
	}

	const iterator end() const
	{
		return EntityView<T>::iterator(*this, size());
	}

  private:
	class iterator
	{
	  public:
		explicit iterator(EntityView<T> &view, size_t index = 0) :
		    m_view(view), m_index(index)
		{
		}

		Entity operator*() const
		{
			return m_view[m_index];
		}

		iterator &operator++()
		{
			m_index++;
			return *this;
		}

		iterator operator++(int)
		{
			return ++(*this);
		}

		bool operator!=(const iterator &rhs) const
		{
			return m_index != rhs.m_index;
		}

	  private:
		size_t         m_index = 0;
		EntityView<T> &m_view;
	};

  private:
	entt::basic_view<entt::entity, entt::exclude_t<>, T> m_view;
};

class EntityManager
{
  public:
	EntityManager() = default;

	~EntityManager();

	Entity create(const std::string &name = "Untitled Entity");

	void clear();

	entt::registry &getRegistry();

  private:
	entt::registry m_registry;
};
}        // namespace Ilum