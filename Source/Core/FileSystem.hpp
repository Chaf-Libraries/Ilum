#pragma once

#include <string>
#include <vector>

namespace Ilum::Core
{
class FileSystem
{
  public:
	static bool IsExist(const std::string &path);
	static bool IsFile(const std::string &path);
	static bool IsDirectory(const std::string &path);
	static bool CreatePath(const std::string &path);
	static bool DeletePath(const std::string &path);
	static bool Copy(const std::string &src, const std::string &dst);

	static const std::string GetFileName(const std::string &path, bool has_extension = true);
	static const std::string GetFileDirectory(const std::string &path);
	static const std::string GetFileExtension(const std::string &path);
	static const std::string GetRelativePath(const std::string &path);

	static bool Save(const std::string &path, const std::vector<uint8_t> &data, bool binary = false);
	static bool Read(const std::string &path, std::vector<uint8_t> &data, bool binary = false, uint32_t begin = 0, uint32_t end = 0);

	static std::string              Toupper(const std::string &str);
	static std::vector<std::string> Split(const std::string &str, char delim);
};
}        // namespace Ilum::Core