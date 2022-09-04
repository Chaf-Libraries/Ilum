#pragma once

#include <CPP14ParserBaseVisitor.h>

#include "Meta.hpp"

#include <iostream>

namespace Ilum
{
class TreeShapeVisitor : public CPP14ParserBaseVisitor
{
  public:
	const std::vector<Meta::TypeMeta> &GetTypeMeta() const;

  public:
	antlrcpp::Any visitChildren(antlr4::tree::ParseTree *node) override;
	antlrcpp::Any visitAttributeSpecifierSeq(CPP14Parser::AttributeSpecifierSeqContext *ctx) override;
	antlrcpp::Any visitAttributeSpecifier(CPP14Parser::AttributeSpecifierContext *ctx) override;
	antlrcpp::Any visitAttribute(CPP14Parser::AttributeContext *ctx) override;
	antlrcpp::Any visitTypeParameter(CPP14Parser::TypeParameterContext *ctx) override;
	antlrcpp::Any visitPointerAbstractDeclarator(CPP14Parser::PointerAbstractDeclaratorContext *ctx) override;
	antlrcpp::Any visitAbstractPackDeclarator(CPP14Parser::AbstractPackDeclaratorContext *ctx) override;
	antlrcpp::Any visitParameterDeclaration(CPP14Parser::ParameterDeclarationContext *ctx) override;
	antlrcpp::Any visitEnumSpecifier(CPP14Parser::EnumSpecifierContext *ctx) override;
	antlrcpp::Any visitEnumeratorList(CPP14Parser::EnumeratorListContext *ctx) override;
	antlrcpp::Any visitEnumeratorDefinition(CPP14Parser::EnumeratorDefinitionContext *ctx) override;
	antlrcpp::Any visitTemplateDeclaration(CPP14Parser::TemplateDeclarationContext *ctx) override;
	antlrcpp::Any visitTemplateparameterList(CPP14Parser::TemplateparameterListContext *ctx) override;
	antlrcpp::Any visitClassSpecifier(CPP14Parser::ClassSpecifierContext *ctx) override;
	antlrcpp::Any visitBaseClause(CPP14Parser::BaseClauseContext *ctx) override;
	antlrcpp::Any visitAccessSpecifier(CPP14Parser::AccessSpecifierContext *ctx) override;
	antlrcpp::Any visitNamespaceDefinition(CPP14Parser::NamespaceDefinitionContext *ctx) override;
	antlrcpp::Any visitNestedNameSpecifier(CPP14Parser::NestedNameSpecifierContext *ctx) override;
	antlrcpp::Any visitQualifiedNamespaceSpecifier(CPP14Parser::QualifiedNamespaceSpecifierContext *ctx) override;
	antlrcpp::Any visitProDeclSpecifierSeq(CPP14Parser::ProDeclSpecifierSeqContext *ctx) override;
	antlrcpp::Any visitMemberDeclaration(CPP14Parser::MemberDeclarationContext *ctx) override;
	antlrcpp::Any visitMemberDeclaratorList(CPP14Parser::MemberDeclaratorListContext *ctx) override;
	antlrcpp::Any visitMemberDeclarator(CPP14Parser::MemberDeclaratorContext *ctx) override;
	antlrcpp::Any visitPointerDeclarator(CPP14Parser::PointerDeclaratorContext *ctx) override;
	antlrcpp::Any visitNoPointerDeclarator(CPP14Parser::NoPointerDeclaratorContext *ctx) override;
	antlrcpp::Any visitBraceOrEqualInitializer(CPP14Parser::BraceOrEqualInitializerContext *ctx) override;
	antlrcpp::Any visitSimpleDeclaration(CPP14Parser::SimpleDeclarationContext *ctx) override;
	antlrcpp::Any visitInitDeclaratorList(CPP14Parser::InitDeclaratorListContext *ctx) override;
	antlrcpp::Any visitInitDeclarator(CPP14Parser::InitDeclaratorContext *ctx) override;
	antlrcpp::Any visitFunctionDefinition(CPP14Parser::FunctionDefinitionContext *ctx) override;

  private:
	std::string GetTextPro(antlr4::ParserRuleContext *ctx) const;

  private:
	std::string_view                   m_code;
	std::vector<Meta::AccessSpecifier> m_access_specifiers;
	std::vector<Meta::TypeMeta>        m_type_metas;
	std::vector<std::string>           m_namespaces;
	std::vector<Meta::Parameter>       m_template_parameters;
	bool                               m_in_template_type = false;
};

}        // namespace Ilum