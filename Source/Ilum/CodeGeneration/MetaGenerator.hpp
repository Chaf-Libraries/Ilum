#pragma once

#include <CPP14ParserBaseVisitor.h>

#include <iostream>

namespace Ilum
{
class TreeShapeVisitor : public CPP14ParserBaseVisitor
{
  public:
	virtual std::any visitNamespaceDefinition(CPP14Parser::NamespaceDefinitionContext *ctx) override
	{
		for (auto& ns : m_namespaces)
		{
			std::cout << ns << " ";
		}
		std::cout << std::endl;

		const size_t origin_size = m_namespaces.size();

		if (ctx->Identifier())
		{
			m_namespaces.push_back(ctx->Identifier()->getText());
		}
		else if (ctx->originalNamespaceName())
		{
			std::vector<std::string> current_namespaces = std::any_cast<std::vector<std::string>>(visitOriginalNamespaceName(ctx->originalNamespaceName()));
			for (auto &current_namespace : current_namespaces)
			{
				m_namespaces.push_back(std::move(current_namespace));
			}
		}

		visit(ctx->declarationseq());

		while (m_namespaces.size()>origin_size)
		{
			m_namespaces.pop_back();
		}

		return {};
	}

	virtual std::any visitEnumSpecifier(CPP14Parser::EnumSpecifierContext *ctx) override final
	{
		auto a = ctx->toString();
		return visitChildren(ctx);
	}

  private:
	std::vector<std::string> m_namespaces;
};

}        // namespace Ilum