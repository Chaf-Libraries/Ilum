#pragma once

#include "Utils/PCH.hpp"

namespace Ilum
{
class FileSystem
{
  public:
	static bool isExist(const std::string &path);
	static bool isFile(const std::string &path);
	static bool isDirectory(const std::string &path);
	static bool createPath(const std::string &path);
	static bool deletePath(const std::string &path);
	static bool copy(const std::string &src, const std::string &dst);

	static const std::string getFileName(const std::string &path, bool has_extension = true);
	static const std::string getFileDirectory(const std::string &path);
	static const std::string getFileExtension(const std::string &path);

	static bool save(const std::string &path, const std::vector<uint8_t> &data, bool binary = false);
	static bool read(const std::string &path, std::vector<uint8_t> &data, bool binary = false, uint32_t begin = 0, uint32_t end = 0);

	static std::string toupper(const std::string &str);
	static std::vector<std::string> split(const std::string &str, char delim);
};
}        // namespace Ilum