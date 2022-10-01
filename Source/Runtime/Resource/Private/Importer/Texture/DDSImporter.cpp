#include "DDSImporter.hpp"

#include <Core/Path.hpp>

#include <dxgiformat.h>

#include <stb_image.h>

namespace Ilum
{
// reference: https://github.com/microsoft/DirectX-Graphics-Samples/blob/master/MiniEngine/Core/dds.h
const uint32_t DDS_MAGIC = 0x20534444;        // "DDS "

inline static std::unordered_map<DXGI_FORMAT, RHIFormat> FormatMap = {
    {DXGI_FORMAT_UNKNOWN, RHIFormat::Undefined},
    {DXGI_FORMAT_R8G8B8A8_UNORM, RHIFormat::R8G8B8A8_UNORM},
    {DXGI_FORMAT_R16_UINT, RHIFormat::R16_UINT},
    {DXGI_FORMAT_R16_SINT, RHIFormat::R16_SINT},
    {DXGI_FORMAT_R16_FLOAT, RHIFormat::R16_FLOAT},
    {DXGI_FORMAT_R16G16_UINT, RHIFormat::R16G16_UINT},
    {DXGI_FORMAT_R16G16_SINT, RHIFormat::R16G16_SINT},
    {DXGI_FORMAT_R16G16_FLOAT, RHIFormat::R16G16_FLOAT},
    {DXGI_FORMAT_R16G16B16A16_UINT, RHIFormat::R16G16B16A16_UINT},
    {DXGI_FORMAT_R16G16B16A16_SINT, RHIFormat::R16G16B16A16_SINT},
    {DXGI_FORMAT_R16G16B16A16_FLOAT, RHIFormat::R16G16B16A16_FLOAT},
    {DXGI_FORMAT_R10G10B10A2_UNORM, RHIFormat::R10G10B10A2_UNORM},
    {DXGI_FORMAT_R10G10B10A2_UINT, RHIFormat::R10G10B10A2_UINT},
    {DXGI_FORMAT_R11G11B10_FLOAT, RHIFormat::R11G11B10_FLOAT},
    {DXGI_FORMAT_R32_UINT, RHIFormat::R32_UINT},
    {DXGI_FORMAT_R32_SINT, RHIFormat::R32_SINT},
    {DXGI_FORMAT_R32_FLOAT, RHIFormat::R32_FLOAT},
    {DXGI_FORMAT_R32G32_UINT, RHIFormat::R32G32_UINT},
    {DXGI_FORMAT_R32G32_SINT, RHIFormat::R32G32_SINT},
    {DXGI_FORMAT_R32G32_FLOAT, RHIFormat::R32G32_FLOAT},
    {DXGI_FORMAT_R32G32B32_UINT, RHIFormat::R32G32B32_UINT},
    {DXGI_FORMAT_R32G32B32_SINT, RHIFormat::R32G32B32_SINT},
    {DXGI_FORMAT_R32G32B32_FLOAT, RHIFormat::R32G32B32_FLOAT},
    {DXGI_FORMAT_R32G32B32A32_UINT, RHIFormat::R32G32B32A32_UINT},
    {DXGI_FORMAT_R32G32B32A32_SINT, RHIFormat::R32G32B32A32_SINT},
    {DXGI_FORMAT_R32G32B32A32_FLOAT, RHIFormat::R32G32B32A32_FLOAT},
    {DXGI_FORMAT_D32_FLOAT, RHIFormat::D32_FLOAT},
    {DXGI_FORMAT_D24_UNORM_S8_UINT, RHIFormat::D24_UNORM_S8_UINT},
};

struct DDS_PIXELFORMAT
{
	uint32_t size;
	uint32_t flags;
	uint32_t fourCC;
	uint32_t RGBBitCount;
	uint32_t RBitMask;
	uint32_t GBitMask;
	uint32_t BBitMask;
	uint32_t ABitMask;
};

#define DDS_FOURCC 0x00000004            // DDPF_FOURCC
#define DDS_RGB 0x00000040               // DDPF_RGB
#define DDS_RGBA 0x00000041              // DDPF_RGB | DDPF_ALPHAPIXELS
#define DDS_LUMINANCE 0x00020000         // DDPF_LUMINANCE
#define DDS_LUMINANCEA 0x00020001        // DDPF_LUMINANCE | DDPF_ALPHAPIXELS
#define DDS_ALPHA 0x00000002             // DDPF_ALPHA
#define DDS_PAL8 0x00000020              // DDPF_PALETTEINDEXED8

#ifndef MAKEFOURCC
#	define MAKEFOURCC(ch0, ch1, ch2, ch3)                                \
		((uint32_t) (uint8_t) (ch0) | ((uint32_t) (uint8_t) (ch1) << 8) | \
		 ((uint32_t) (uint8_t) (ch2) << 16) | ((uint32_t) (uint8_t) (ch3) << 24))
#endif /* defined(MAKEFOURCC) */

extern __declspec(selectany) const DDS_PIXELFORMAT DDSPF_DXT1 =
    {sizeof(DDS_PIXELFORMAT), DDS_FOURCC, MAKEFOURCC('D', 'X', 'T', '1'), 0, 0, 0, 0, 0};

extern __declspec(selectany) const DDS_PIXELFORMAT DDSPF_DXT2 =
    {sizeof(DDS_PIXELFORMAT), DDS_FOURCC, MAKEFOURCC('D', 'X', 'T', '2'), 0, 0, 0, 0, 0};

extern __declspec(selectany) const DDS_PIXELFORMAT DDSPF_DXT3 =
    {sizeof(DDS_PIXELFORMAT), DDS_FOURCC, MAKEFOURCC('D', 'X', 'T', '3'), 0, 0, 0, 0, 0};

extern __declspec(selectany) const DDS_PIXELFORMAT DDSPF_DXT4 =
    {sizeof(DDS_PIXELFORMAT), DDS_FOURCC, MAKEFOURCC('D', 'X', 'T', '4'), 0, 0, 0, 0, 0};

extern __declspec(selectany) const DDS_PIXELFORMAT DDSPF_DXT5 =
    {sizeof(DDS_PIXELFORMAT), DDS_FOURCC, MAKEFOURCC('D', 'X', 'T', '5'), 0, 0, 0, 0, 0};

extern __declspec(selectany) const DDS_PIXELFORMAT DDSPF_BC4_UNORM =
    {sizeof(DDS_PIXELFORMAT), DDS_FOURCC, MAKEFOURCC('B', 'C', '4', 'U'), 0, 0, 0, 0, 0};

extern __declspec(selectany) const DDS_PIXELFORMAT DDSPF_BC4_SNORM =
    {sizeof(DDS_PIXELFORMAT), DDS_FOURCC, MAKEFOURCC('B', 'C', '4', 'S'), 0, 0, 0, 0, 0};

extern __declspec(selectany) const DDS_PIXELFORMAT DDSPF_BC5_UNORM =
    {sizeof(DDS_PIXELFORMAT), DDS_FOURCC, MAKEFOURCC('B', 'C', '5', 'U'), 0, 0, 0, 0, 0};

extern __declspec(selectany) const DDS_PIXELFORMAT DDSPF_BC5_SNORM =
    {sizeof(DDS_PIXELFORMAT), DDS_FOURCC, MAKEFOURCC('B', 'C', '5', 'S'), 0, 0, 0, 0, 0};

extern __declspec(selectany) const DDS_PIXELFORMAT DDSPF_R8G8_B8G8 =
    {sizeof(DDS_PIXELFORMAT), DDS_FOURCC, MAKEFOURCC('R', 'G', 'B', 'G'), 0, 0, 0, 0, 0};

extern __declspec(selectany) const DDS_PIXELFORMAT DDSPF_G8R8_G8B8 =
    {sizeof(DDS_PIXELFORMAT), DDS_FOURCC, MAKEFOURCC('G', 'R', 'G', 'B'), 0, 0, 0, 0, 0};

extern __declspec(selectany) const DDS_PIXELFORMAT DDSPF_YUY2 =
    {sizeof(DDS_PIXELFORMAT), DDS_FOURCC, MAKEFOURCC('Y', 'U', 'Y', '2'), 0, 0, 0, 0, 0};

extern __declspec(selectany) const DDS_PIXELFORMAT DDSPF_A8R8G8B8 =
    {sizeof(DDS_PIXELFORMAT), DDS_RGBA, 0, 32, 0x00ff0000, 0x0000ff00, 0x000000ff, 0xff000000};

extern __declspec(selectany) const DDS_PIXELFORMAT DDSPF_X8R8G8B8 =
    {sizeof(DDS_PIXELFORMAT), DDS_RGB, 0, 32, 0x00ff0000, 0x0000ff00, 0x000000ff, 0x00000000};

extern __declspec(selectany) const DDS_PIXELFORMAT DDSPF_A8B8G8R8 =
    {sizeof(DDS_PIXELFORMAT), DDS_RGBA, 0, 32, 0x000000ff, 0x0000ff00, 0x00ff0000, 0xff000000};

extern __declspec(selectany) const DDS_PIXELFORMAT DDSPF_X8B8G8R8 =
    {sizeof(DDS_PIXELFORMAT), DDS_RGB, 0, 32, 0x000000ff, 0x0000ff00, 0x00ff0000, 0x00000000};

extern __declspec(selectany) const DDS_PIXELFORMAT DDSPF_G16R16 =
    {sizeof(DDS_PIXELFORMAT), DDS_RGB, 0, 32, 0x0000ffff, 0xffff0000, 0x00000000, 0x00000000};

extern __declspec(selectany) const DDS_PIXELFORMAT DDSPF_R5G6B5 =
    {sizeof(DDS_PIXELFORMAT), DDS_RGB, 0, 16, 0x0000f800, 0x000007e0, 0x0000001f, 0x00000000};

extern __declspec(selectany) const DDS_PIXELFORMAT DDSPF_A1R5G5B5 =
    {sizeof(DDS_PIXELFORMAT), DDS_RGBA, 0, 16, 0x00007c00, 0x000003e0, 0x0000001f, 0x00008000};

extern __declspec(selectany) const DDS_PIXELFORMAT DDSPF_A4R4G4B4 =
    {sizeof(DDS_PIXELFORMAT), DDS_RGBA, 0, 16, 0x00000f00, 0x000000f0, 0x0000000f, 0x0000f000};

extern __declspec(selectany) const DDS_PIXELFORMAT DDSPF_R8G8B8 =
    {sizeof(DDS_PIXELFORMAT), DDS_RGB, 0, 24, 0x00ff0000, 0x0000ff00, 0x000000ff, 0x00000000};

extern __declspec(selectany) const DDS_PIXELFORMAT DDSPF_L8 =
    {sizeof(DDS_PIXELFORMAT), DDS_LUMINANCE, 0, 8, 0xff, 0x00, 0x00, 0x00};

extern __declspec(selectany) const DDS_PIXELFORMAT DDSPF_L16 =
    {sizeof(DDS_PIXELFORMAT), DDS_LUMINANCE, 0, 16, 0xffff, 0x0000, 0x0000, 0x0000};

extern __declspec(selectany) const DDS_PIXELFORMAT DDSPF_A8L8 =
    {sizeof(DDS_PIXELFORMAT), DDS_LUMINANCEA, 0, 16, 0x00ff, 0x0000, 0x0000, 0xff00};

extern __declspec(selectany) const DDS_PIXELFORMAT DDSPF_A8 =
    {sizeof(DDS_PIXELFORMAT), DDS_ALPHA, 0, 8, 0x00, 0x00, 0x00, 0xff};

// D3DFMT_A2R10G10B10/D3DFMT_A2B10G10R10 should be written using DX10 extension to avoid D3DX 10:10:10:2 reversal issue

// This indicates the DDS_HEADER_DXT10 extension is present (the format is in dxgiFormat)
extern __declspec(selectany) const DDS_PIXELFORMAT DDSPF_DX10 =
    {sizeof(DDS_PIXELFORMAT), DDS_FOURCC, MAKEFOURCC('D', 'X', '1', '0'), 0, 0, 0, 0, 0};

#define DDS_HEADER_FLAGS_TEXTURE 0x00001007           // DDSD_CAPS | DDSD_HEIGHT | DDSD_WIDTH | DDSD_PIXELFORMAT
#define DDS_HEADER_FLAGS_MIPMAP 0x00020000            // DDSD_MIPMAPCOUNT
#define DDS_HEADER_FLAGS_VOLUME 0x00800000            // DDSD_DEPTH
#define DDS_HEADER_FLAGS_PITCH 0x00000008             // DDSD_PITCH
#define DDS_HEADER_FLAGS_LINEARSIZE 0x00080000        // DDSD_LINEARSIZE

#define DDS_HEIGHT 0x00000002        // DDSD_HEIGHT
#define DDS_WIDTH 0x00000004         // DDSD_WIDTH

#define DDS_SURFACE_FLAGS_TEXTURE 0x00001000        // DDSCAPS_TEXTURE
#define DDS_SURFACE_FLAGS_MIPMAP 0x00400008         // DDSCAPS_COMPLEX | DDSCAPS_MIPMAP
#define DDS_SURFACE_FLAGS_CUBEMAP 0x00000008        // DDSCAPS_COMPLEX

#define DDS_CUBEMAP_POSITIVEX 0x00000600        // DDSCAPS2_CUBEMAP | DDSCAPS2_CUBEMAP_POSITIVEX
#define DDS_CUBEMAP_NEGATIVEX 0x00000a00        // DDSCAPS2_CUBEMAP | DDSCAPS2_CUBEMAP_NEGATIVEX
#define DDS_CUBEMAP_POSITIVEY 0x00001200        // DDSCAPS2_CUBEMAP | DDSCAPS2_CUBEMAP_POSITIVEY
#define DDS_CUBEMAP_NEGATIVEY 0x00002200        // DDSCAPS2_CUBEMAP | DDSCAPS2_CUBEMAP_NEGATIVEY
#define DDS_CUBEMAP_POSITIVEZ 0x00004200        // DDSCAPS2_CUBEMAP | DDSCAPS2_CUBEMAP_POSITIVEZ
#define DDS_CUBEMAP_NEGATIVEZ 0x00008200        // DDSCAPS2_CUBEMAP | DDSCAPS2_CUBEMAP_NEGATIVEZ

#define DDS_CUBEMAP_ALLFACES (DDS_CUBEMAP_POSITIVEX | DDS_CUBEMAP_NEGATIVEX | \
	                          DDS_CUBEMAP_POSITIVEY | DDS_CUBEMAP_NEGATIVEY | \
	                          DDS_CUBEMAP_POSITIVEZ | DDS_CUBEMAP_NEGATIVEZ)

#define DDS_CUBEMAP 0x00000200        // DDSCAPS2_CUBEMAP

#define DDS_FLAGS_VOLUME 0x00200000        // DDSCAPS2_VOLUME

// Subset here matches D3D10_RESOURCE_DIMENSION and D3D11_RESOURCE_DIMENSION
enum DDS_RESOURCE_DIMENSION
{
	DDS_DIMENSION_TEXTURE1D = 2,
	DDS_DIMENSION_TEXTURE2D = 3,
	DDS_DIMENSION_TEXTURE3D = 4,
};

// Subset here matches D3D10_RESOURCE_MISC_FLAG and D3D11_RESOURCE_MISC_FLAG
enum DDS_RESOURCE_MISC_FLAG
{
	DDS_RESOURCE_MISC_TEXTURECUBE = 0x4L,
};

enum DDS_MISC_FLAGS2
{
	DDS_MISC_FLAGS2_ALPHA_MODE_MASK = 0x7L,
};

struct DDS_HEADER
{
	uint32_t        size;
	uint32_t        flags;
	uint32_t        height;
	uint32_t        width;
	uint32_t        pitchOrLinearSize;
	uint32_t        depth;        // only if DDS_HEADER_FLAGS_VOLUME is set in flags
	uint32_t        mipMapCount;
	uint32_t        reserved1[11];
	DDS_PIXELFORMAT ddspf;
	uint32_t        caps;
	uint32_t        caps2;
	uint32_t        caps3;
	uint32_t        caps4;
	uint32_t        reserved2;
};

struct DDS_HEADER_DXT10
{
	DXGI_FORMAT dxgiFormat;
	uint32_t    resourceDimension;
	uint32_t    miscFlag;        // see D3D11_RESOURCE_MISC_FLAG
	uint32_t    arraySize;
	uint32_t    miscFlags2;        // see DDS_MISC_FLAGS2
};

static_assert(sizeof(DDS_HEADER) == 124, "DDS Header size mismatch");
static_assert(sizeof(DDS_HEADER_DXT10) == 20, "DDS DX10 Extended Header size mismatch");

#define ISBITMASK(r, g, b, a) (ddpf.RBitMask == r && ddpf.GBitMask == g && ddpf.BBitMask == b && ddpf.ABitMask == a)

static DXGI_FORMAT GetDXGIFormat(const DDS_PIXELFORMAT &ddpf)
{
	if (ddpf.flags & DDS_RGB)
	{
		// Note that sRGB formats are written using the "DX10" extended header

		switch (ddpf.RGBBitCount)
		{
			case 32:
				if (ISBITMASK(0x000000ff, 0x0000ff00, 0x00ff0000, 0xff000000))
				{
					return DXGI_FORMAT_R8G8B8A8_UNORM;
				}

				if (ISBITMASK(0x00ff0000, 0x0000ff00, 0x000000ff, 0xff000000))
				{
					return DXGI_FORMAT_B8G8R8A8_UNORM;
				}

				if (ISBITMASK(0x00ff0000, 0x0000ff00, 0x000000ff, 0x00000000))
				{
					return DXGI_FORMAT_B8G8R8X8_UNORM;
				}

				// No DXGI format maps to ISBITMASK(0x000000ff,0x0000ff00,0x00ff0000,0x00000000) aka D3DFMT_X8B8G8R8

				// Note that many common DDS reader/writers (including D3DX) swap the
				// the RED/BLUE masks for 10:10:10:2 formats. We assumme
				// below that the 'backwards' header mask is being used since it is most
				// likely written by D3DX. The more robust solution is to use the 'DX10'
				// header extension and specify the DXGI_FORMAT_R10G10B10A2_UNORM format directly

				// For 'correct' writers, this should be 0x000003ff,0x000ffc00,0x3ff00000 for RGB data
				if (ISBITMASK(0x3ff00000, 0x000ffc00, 0x000003ff, 0xc0000000))
				{
					return DXGI_FORMAT_R10G10B10A2_UNORM;
				}

				// No DXGI format maps to ISBITMASK(0x000003ff,0x000ffc00,0x3ff00000,0xc0000000) aka D3DFMT_A2R10G10B10

				if (ISBITMASK(0x0000ffff, 0xffff0000, 0x00000000, 0x00000000))
				{
					return DXGI_FORMAT_R16G16_UNORM;
				}

				if (ISBITMASK(0xffffffff, 0x00000000, 0x00000000, 0x00000000))
				{
					// Only 32-bit color channel format in D3D9 was R32F
					return DXGI_FORMAT_R32_FLOAT;        // D3DX writes this out as a FourCC of 114
				}
				break;

			case 24:
				// No 24bpp DXGI formats aka D3DFMT_R8G8B8
				break;

			case 16:
				if (ISBITMASK(0x7c00, 0x03e0, 0x001f, 0x8000))
				{
					return DXGI_FORMAT_B5G5R5A1_UNORM;
				}
				if (ISBITMASK(0xf800, 0x07e0, 0x001f, 0x0000))
				{
					return DXGI_FORMAT_B5G6R5_UNORM;
				}

				// No DXGI format maps to ISBITMASK(0x7c00,0x03e0,0x001f,0x0000) aka D3DFMT_X1R5G5B5

				if (ISBITMASK(0x0f00, 0x00f0, 0x000f, 0xf000))
				{
					return DXGI_FORMAT_B4G4R4A4_UNORM;
				}

				// No DXGI format maps to ISBITMASK(0x0f00,0x00f0,0x000f,0x0000) aka D3DFMT_X4R4G4B4

				// No 3:3:2, 3:3:2:8, or paletted DXGI formats aka D3DFMT_A8R3G3B2, D3DFMT_R3G3B2, D3DFMT_P8, D3DFMT_A8P8, etc.
				break;
		}
	}
	else if (ddpf.flags & DDS_LUMINANCE)
	{
		if (8 == ddpf.RGBBitCount)
		{
			if (ISBITMASK(0x000000ff, 0x00000000, 0x00000000, 0x00000000))
			{
				return DXGI_FORMAT_R8_UNORM;        // D3DX10/11 writes this out as DX10 extension
			}

			// No DXGI format maps to ISBITMASK(0x0f,0x00,0x00,0xf0) aka D3DFMT_A4L4
		}

		if (16 == ddpf.RGBBitCount)
		{
			if (ISBITMASK(0x0000ffff, 0x00000000, 0x00000000, 0x00000000))
			{
				return DXGI_FORMAT_R16_UNORM;        // D3DX10/11 writes this out as DX10 extension
			}
			if (ISBITMASK(0x000000ff, 0x00000000, 0x00000000, 0x0000ff00))
			{
				return DXGI_FORMAT_R8G8_UNORM;        // D3DX10/11 writes this out as DX10 extension
			}
		}
	}
	else if (ddpf.flags & DDS_ALPHA)
	{
		if (8 == ddpf.RGBBitCount)
		{
			return DXGI_FORMAT_A8_UNORM;
		}
	}
	else if (ddpf.flags & DDS_FOURCC)
	{
		if (MAKEFOURCC('D', 'X', 'T', '1') == ddpf.fourCC)
		{
			return DXGI_FORMAT_BC1_UNORM;
		}
		if (MAKEFOURCC('D', 'X', 'T', '3') == ddpf.fourCC)
		{
			return DXGI_FORMAT_BC2_UNORM;
		}
		if (MAKEFOURCC('D', 'X', 'T', '5') == ddpf.fourCC)
		{
			return DXGI_FORMAT_BC3_UNORM;
		}

		// While pre-mulitplied alpha isn't directly supported by the DXGI formats,
		// they are basically the same as these BC formats so they can be mapped
		if (MAKEFOURCC('D', 'X', 'T', '2') == ddpf.fourCC)
		{
			return DXGI_FORMAT_BC2_UNORM;
		}
		if (MAKEFOURCC('D', 'X', 'T', '4') == ddpf.fourCC)
		{
			return DXGI_FORMAT_BC3_UNORM;
		}

		if (MAKEFOURCC('A', 'T', 'I', '1') == ddpf.fourCC)
		{
			return DXGI_FORMAT_BC4_UNORM;
		}
		if (MAKEFOURCC('B', 'C', '4', 'U') == ddpf.fourCC)
		{
			return DXGI_FORMAT_BC4_UNORM;
		}
		if (MAKEFOURCC('B', 'C', '4', 'S') == ddpf.fourCC)
		{
			return DXGI_FORMAT_BC4_SNORM;
		}

		if (MAKEFOURCC('A', 'T', 'I', '2') == ddpf.fourCC)
		{
			return DXGI_FORMAT_BC5_UNORM;
		}
		if (MAKEFOURCC('B', 'C', '5', 'U') == ddpf.fourCC)
		{
			return DXGI_FORMAT_BC5_UNORM;
		}
		if (MAKEFOURCC('B', 'C', '5', 'S') == ddpf.fourCC)
		{
			return DXGI_FORMAT_BC5_SNORM;
		}

		// BC6H and BC7 are written using the "DX10" extended header

		if (MAKEFOURCC('R', 'G', 'B', 'G') == ddpf.fourCC)
		{
			return DXGI_FORMAT_R8G8_B8G8_UNORM;
		}
		if (MAKEFOURCC('G', 'R', 'G', 'B') == ddpf.fourCC)
		{
			return DXGI_FORMAT_G8R8_G8B8_UNORM;
		}

		if (MAKEFOURCC('Y', 'U', 'Y', '2') == ddpf.fourCC)
		{
			return DXGI_FORMAT_YUY2;
		}

		// Check for D3DFORMAT enums being set here
		switch (ddpf.fourCC)
		{
			case 36:        // D3DFMT_A16B16G16R16
				return DXGI_FORMAT_R16G16B16A16_UNORM;

			case 110:        // D3DFMT_Q16W16V16U16
				return DXGI_FORMAT_R16G16B16A16_SNORM;

			case 111:        // D3DFMT_R16F
				return DXGI_FORMAT_R16_FLOAT;

			case 112:        // D3DFMT_G16R16F
				return DXGI_FORMAT_R16G16_FLOAT;

			case 113:        // D3DFMT_A16B16G16R16F
				return DXGI_FORMAT_R16G16B16A16_FLOAT;

			case 114:        // D3DFMT_R32F
				return DXGI_FORMAT_R32_FLOAT;

			case 115:        // D3DFMT_G32R32F
				return DXGI_FORMAT_R32G32_FLOAT;

			case 116:        // D3DFMT_A32B32G32R32F
				return DXGI_FORMAT_R32G32B32A32_FLOAT;
		}
	}

	return DXGI_FORMAT_UNKNOWN;
}

TextureImportInfo DDSImporter::ImportImpl(const std::string &filename)
{
	TextureImportInfo info = {};
	info.desc.name         = Path::GetInstance().GetFileName(filename, false);
	info.desc.width        = 1;
	info.desc.height       = 1;
	info.desc.depth        = 1;
	info.desc.mips         = 1;
	info.desc.layers       = 1;
	info.desc.samples      = 1;
	info.desc.external     = true;

	std::vector<uint8_t> raw_data;
	Path::GetInstance().Read(filename, raw_data, true);

	size_t   data_size = raw_data.size();
	uint8_t *data      = raw_data.data();

	uint32_t dwMagicNumber = *(const uint32_t *) (data);
	if (dwMagicNumber != DDS_MAGIC)
	{
		return {};
	}

	auto header = reinterpret_cast<const DDS_HEADER *>(data + sizeof(uint32_t));
	if (header->size != sizeof(DDS_HEADER) ||
	    header->ddspf.size != sizeof(DDS_PIXELFORMAT))
	{
		return {};
	}

	size_t offset = sizeof(DDS_HEADER) + sizeof(uint32_t);

	// Check for extensions
	if (header->ddspf.flags & DDS_FOURCC)
	{
		if (MAKEFOURCC('D', 'X', '1', '0') == header->ddspf.fourCC)
		{
			offset += sizeof(DDS_HEADER_DXT10);
		}
	}

	if (data_size < offset)
	{
		return {};
	}

	info.desc.width  = header->width;
	info.desc.height = header->height;
	info.desc.depth  = header->depth;

	if ((header->ddspf.flags & DDS_FOURCC) && (MAKEFOURCC('D', 'X', '1', '0') == header->ddspf.fourCC))
	{
		auto d3d10ext = reinterpret_cast<const DDS_HEADER_DXT10 *>((const char *) header + sizeof(DDS_HEADER));

		info.desc.layers = d3d10ext->arraySize;
		if (info.desc.layers == 0)
		{
			return {};
		}

		if (FormatMap.find(d3d10ext->dxgiFormat) != FormatMap.end())
		{
			info.desc.format = FormatMap[d3d10ext->dxgiFormat];
		}
		else
		{
			return {};
		}
	}
	else
	{
		auto dxgi_format = GetDXGIFormat(header->ddspf);
		if (FormatMap.find(dxgi_format) != FormatMap.end())
		{
			info.desc.format = FormatMap[dxgi_format];
		}
		else
		{
			return {};
		}

		if (header->flags & DDS_HEADER_FLAGS_VOLUME)
		{
			// So it's a 3D texture
		}
		else
		{
			info.desc.depth  = 1;
			info.desc.layers = 1;

			if (header->caps2 & DDS_CUBEMAP)
			{
				// Require all six faces to be defined
				if ((header->caps2 & DDS_CUBEMAP_ALLFACES) != DDS_CUBEMAP_ALLFACES)
				{
					return {};
				}

				info.desc.layers = 6;
			}
		}
	}

	uint8_t *tex_data = data + offset;
	data_size -= offset;

	info.data.resize(data_size);
	std::memcpy(info.data.data(), tex_data, data_size);

	info.desc.mips  = static_cast<uint32_t>(std::floor(std::log2(std::max(info.desc.width, info.desc.height))) + 1);
	info.desc.usage = RHITextureUsage::ShaderResource | RHITextureUsage::Transfer;

	return info;
}
}        // namespace Ilum