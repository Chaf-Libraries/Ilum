#pragma once

#include "ThreadPool.hpp"

namespace Ilum
{
template <typename T>
bool TQueue<T>::push(const T &value)
{
	std::unique_lock<std::mutex> lock(m_mutex);
	this->m_queue.push(value);
	return true;
}

template <typename T>
bool TQueue<T>::pop(T &value)
{
	std::unique_lock<std::mutex> lock(m_mutex);
	if (m_queue.empty())
	{
		return false;
	}
	value = m_queue.front();
	m_queue.pop();
	return true;
}

template <typename T>
bool TQueue<T>::empty()
{
	std::unique_lock<std::mutex> lock(m_mutex);
	return m_queue.empty();
}

template <typename F, typename... Args>
auto ThreadPool::addTask(F &&f, Args &&...args) -> std::future<decltype(f(0, args...))>
{
	auto pack = std::make_shared<std::packaged_task<decltype(f(0, args...))(size_t)>>(std::bind(std::forward<F>(f), std::placeholders::_1, std::forward<Args>(args)...));
	auto func = new std::function<void(size_t id)>([pack](size_t id) { (*pack)(id); });
	m_queue.push(func);
	std::unique_lock<std::mutex> lock(m_mutex);
	m_condition.notify_one();
	return pack->get_future();
}

template <typename F>
auto ThreadPool::addTask(F &&f) -> std::future<decltype(f(0))>
{
	auto pack = std::make_shared<std::packaged_task<decltype(f(0))(size_t)>>(std::forward<F>(f));
	auto func = new std::function<void(size_t id)>([pack](size_t id) { (*pack)(id); });
	m_queue.push(func);
	std::unique_lock<std::mutex> lock(m_mutex);
	m_condition.notify_one();
	return pack->get_future();
}
}        // namespace Ilum