#pragma once

namespace Ilum
{
enum class PinType : uint64_t
{
	None = 0,

	Scalar = 1 << 0,
	Vec2 = 1 << 1,
	Vec3 = 1 << 2,
	Vec4 = 1 << 3,

	Texture2D      = 1 << 4,
	Texture2DArray = 1 << 5,
	Texture3D      = 1 << 6,
	Texture3DArray = 1 << 7,
	TextureCube    = 1 << 8,
	TextureCubeArray = 1 << 9,

	SamplerState = 1 << 10,

	BxDF = 1 << 11,
};

inline PinType operator|(PinType lhs, PinType rhs)
{
	return (PinType) ((uint64_t) lhs | (uint64_t) rhs);
}

inline bool operator&(PinType lhs, PinType rhs)
{
	return (uint64_t) lhs & (uint64_t) rhs;
}
}        // namespace Ilum