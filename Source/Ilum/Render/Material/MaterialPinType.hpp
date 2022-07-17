#pragma once

namespace Ilum
{
enum class PinType
{
	None,

	Float,
	Half,
	Double,
	Int,
	Uint,
	Bool,

	Float2,
	Half2,
	Double2,
	Int2,
	Uint2,
	Bool2,

	Float3,
	Half3,
	Double3,
	Int3,
	Uint3,
	Bool3,

	Float4,
	Half4,
	Double4,
	Int4,
	Uint4,
	Bool4,

	Texture2D,
	Texture2DArray,
	Texture3D,
	Texture3DArray,
	TextureCube,
	TextureCubeArray,

	RWTexture2D,
	RWTexture2DArray,
	RWTexture3D,
	RWTexture3DArray,
	RWTextureCube,
	RWTextureCubeArray,

	SamplerState,

	BxDF,
};
}