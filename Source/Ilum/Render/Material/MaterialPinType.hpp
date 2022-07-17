#pragma once

namespace Ilum
{
enum class PinType : uint64_t
{
	None = 0,

	Float = 1 << 0,
	Half  = 1 << 0,
	Double = 1 << 0,
	Int    = 1 << 0,
	Uint   = 1 << 0,
	Bool   = 1 << 0,

	Float2 = 1 << 0,
	Half2  = 1 << 0,
	Double2 = 1 << 0,
	Int2    = 1 << 0,
	Uint2   = 1 << 0,
	Bool2   = 1 << 0,

	Float3 = 1 << 0,
	Half3  = 1 << 0,
	Double3 = 1 << 0,
	Int3    = 1 << 0,
	Uint3   = 1 << 0,
	Bool3   = 1 << 0,

	Float4 = 1 << 0,
	Half4  = 1 << 0,
	Double4 = 1 << 0,
	Int4    = 1 << 0,
	Uint4   = 1 << 0,
	Bool4   = 1 << 0,

	Texture2D      = 1 << 0,
	Texture2DArray = 1 << 0,
	Texture3D      = 1 << 0,
	Texture3DArray = 1 << 0,
	TextureCube    = 1 << 0,
	TextureCubeArray = 1 << 0,

	RWTexture2D = 1 << 0,
	RWTexture2DArray = 1 << 0,
	RWTexture3D      = 1 << 0,
	RWTexture3DArray = 1 << 0,
	RWTextureCube    = 1 << 0,
	RWTextureCubeArray = 1 << 0,

	SamplerState = 1 << 0,

	BxDF = 1 << 0,
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