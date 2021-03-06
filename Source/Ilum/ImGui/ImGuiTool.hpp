#pragma once

#include <imgui.h>

#include <iostream>

static inline ImVec2 operator*(const ImVec2 &lhs, const float rhs)
{
	return ImVec2(lhs.x * rhs, lhs.y * rhs);
}

static inline ImVec2 operator/(const ImVec2 &lhs, const float rhs)
{
	return ImVec2(lhs.x / rhs, lhs.y / rhs);
}

static inline ImVec2 operator+(const ImVec2 &lhs, const ImVec2 &rhs)
{
	return ImVec2(lhs.x + rhs.x, lhs.y + rhs.y);
}

static inline ImVec2 operator-(const ImVec2 &lhs, const ImVec2 &rhs)
{
	return ImVec2(lhs.x - rhs.x, lhs.y - rhs.y);
}

static inline ImVec2 operator*(const ImVec2 &lhs, const ImVec2 &rhs)
{
	return ImVec2(lhs.x * rhs.x, lhs.y * rhs.y);
}

static inline ImVec2 operator/(const ImVec2 &lhs, const ImVec2 &rhs)
{
	return ImVec2(lhs.x / rhs.x, lhs.y / rhs.y);
}

static inline ImVec2 &operator+=(ImVec2 &lhs, const ImVec2 &rhs)
{
	lhs.x += rhs.x;
	lhs.y += rhs.y;
	return lhs;
}

static inline ImVec2 &operator-=(ImVec2 &lhs, const ImVec2 &rhs)
{
	lhs.x -= rhs.x;
	lhs.y -= rhs.y;
	return lhs;
}

static inline ImVec2 &operator*=(ImVec2 &lhs, const float rhs)
{
	lhs.x *= rhs;
	lhs.y *= rhs;
	return lhs;
}

static inline ImVec2 &operator/=(ImVec2 &lhs, const float rhs)
{
	lhs.x /= rhs;
	lhs.y /= rhs;
	return lhs;
}

static inline ImVec4 operator-(const ImVec4 &lhs, const ImVec4 &rhs)
{
	return ImVec4(lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z, lhs.w - rhs.w);
}

static inline ImVec4 operator+(const ImVec4 &lhs, const ImVec4 &rhs)
{
	return ImVec4(lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z, lhs.w + rhs.w);
}

static inline ImVec4 operator*(const ImVec4 &lhs, const float rhs)
{
	return ImVec4(lhs.x * rhs, lhs.y * rhs, lhs.z * rhs, lhs.w * rhs);
}

static inline ImVec4 operator/(const ImVec4 &lhs, const float rhs)
{
	return ImVec4(lhs.x / rhs, lhs.y / rhs, lhs.z / rhs, lhs.w / rhs);
}

static inline ImVec4 operator*(const ImVec4 &lhs, const ImVec4 &rhs)
{
	return ImVec4(lhs.x * rhs.x, lhs.y * rhs.y, lhs.z * rhs.z, lhs.w * rhs.w);
}

static inline ImVec4 operator/(const ImVec4 &lhs, const ImVec4 &rhs)
{
	return ImVec4(lhs.x / rhs.x, lhs.y / rhs.y, lhs.z / rhs.z, lhs.w / rhs.w);
}

static inline ImVec4 &operator+=(ImVec4 &lhs, const ImVec4 &rhs)
{
	lhs.x += rhs.x;
	lhs.y += rhs.y;
	lhs.z += rhs.z;
	lhs.w += rhs.w;
	return lhs;
}

static inline ImVec4 &operator-=(ImVec4 &lhs, const ImVec4 &rhs)
{
	lhs.x -= rhs.x;
	lhs.y -= rhs.y;
	lhs.z -= rhs.z;
	lhs.w -= rhs.w;
	return lhs;
}

static inline ImVec4 &operator*=(ImVec4 &lhs, const float rhs)
{
	lhs.x *= rhs;
	lhs.y *= rhs;
	return lhs;
}

static inline ImVec4 &operator/=(ImVec4 &lhs, const float rhs)
{
	lhs.x /= rhs;
	lhs.y /= rhs;
	return lhs;
}

static inline std::ostream &operator<<(std::ostream &ostream, const ImVec2 a)
{
	ostream << "{ " << a.x << ", " << a.y << " }";
	return ostream;
}

static inline std::ostream &operator<<(std::ostream &ostream, const ImVec4 a)
{
	ostream << "{ " << a.x << ", " << a.y << ", " << a.z << ", " << a.w << " }";
	return ostream;
}