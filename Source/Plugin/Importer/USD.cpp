#include <Resource/Importer.hpp>

using namespace Ilum;

class USDImporter : public Importer<ResourceType::Prefab>
{
  public:
  protected:
	virtual void Import_(ResourceManager *manager, const std::string &path, RHIContext *rhi_context) override
	{
	}
};

extern "C"
{
	EXPORT_API USDImporter *Create()
	{
		return new USDImporter;
	}
}