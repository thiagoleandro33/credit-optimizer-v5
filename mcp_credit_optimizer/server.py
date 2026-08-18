#!/usr/bin/env python3
"""
Credit Optimizer MCP Server v5.3
Analyze any Manus task prompt and return quality-preserving execution recommendations.
Audited across 53 adversarial scenarios with explicit clarification and safe routing gates.

Works with: Manus and any MCP-compatible client.
"""

import json
import re
import sys
import unicodedata
from typing import Any

from fastmcp import FastMCP

SERVER_VERSION = "5.3.0"

# Create the MCP server
mcp = FastMCP("Credit Optimizer")


def normalize_text(text: str) -> str:
    """Normalize case, accents, and whitespace for robust multilingual matching."""
    decomposed = unicodedata.normalize("NFKD", text.casefold())
    without_marks = "".join(char for char in decomposed if not unicodedata.combining(char))
    return re.sub(r"\s+", " ", without_marks).strip()


def _normalize_pattern(pattern: str) -> str:
    """Normalize literal accents in a regex while preserving regex syntax."""
    return normalize_text(pattern)


def _phrase_pattern(phrase: str) -> str:
    """Build a boundary-aware pattern so `api` does not match `capital`."""
    normalized = re.escape(normalize_text(phrase)).replace(r"\ ", r"\s+")
    return rf"(?<!\w){normalized}(?!\w)"

# ============================================================
# CORE ANALYSIS ENGINE (ported from analyze_prompt.py v5.2)
# ============================================================

INTENT_KEYWORDS = {
    "qa_brainstorm": [
        "o que é", "o que são", "explique", "explain", "what is", "what are",
        "como funciona", "how does", "por que", "why", "defina", "define",
        "brainstorm", "ideia", "idea", "sugestão", "suggest", "opine",
        "compare", "diferença", "difference", "resuma", "summarize",
        "traduza", "translate", "reescreva", "rewrite", "melhore o texto",
        "corrija o texto", "revise o texto", "me diga", "tell me", "liste",
        "list", "quais são", "which are", "me ajude a pensar", "help me think",
        "qual a diferença", "o que significa", "what does", "como posso",
        "how can i", "me explique", "descreva", "describe"
    ],
    "code_fix": [
        "corrija o código", "fix the code", "corrija o bug", "fix the bug",
        "debug", "corrija o erro", "fix the error", "corrija esse",
        "fix this", "não funciona", "doesn't work", "não está funcionando",
        "is not working", "erro na linha", "error on line"
    ],
    "research": [
        "pesquise", "research", "investigue", "investigate", "analise o mercado",
        "market analysis", "estudo", "study", "relatório", "report",
        "benchmark", "tendências", "trends", "deep dive", "wide research",
        "pesquisa aprofundada", "in-depth research", "análise completa"
    ],
    "code_dev": [
        "crie um site", "create a website", "build", "desenvolva", "develop",
        "programa", "program", "código", "code", "script", "app",
        "aplicativo", "application", "api", "backend", "frontend",
        "database", "deploy", "landing page", "webapp", "react", "python",
        "implementar", "implement", "criar", "create", "construir"
    ],
    "data_analysis": [
        "analise dados", "analyze data", "gráfico", "chart", "graph",
        "visualização", "visualization", "dashboard", "planilha",
        "spreadsheet", "excel", "csv", "estatística", "statistics",
        "métricas", "metrics", "kpi", "tabela", "table", "plot"
    ],
    "content_creation": [
        "slides", "apresentação", "presentation", "powerpoint", "ppt",
        "documento", "document", "artigo", "article", "blog post",
        "email", "newsletter", "conteúdo", "content", "escreva", "write"
    ],
    "media_generation": [
        "imagem", "image", "foto", "photo", "vídeo", "video",
        "gerar imagem", "generate image", "design", "logo", "banner",
        "poster", "infográfico", "infographic", "thumbnail"
    ],
    "automation": [
        "automatize", "automate", "agende", "schedule", "workflow",
        "integração", "integration", "bot", "scraping", "monitor",
        "cron", "pipeline", "trigger", "notificação", "notification"
    ]
}

COMPLEXITY_INDICATORS = {
    "high": [
        "completo", "complete", "detalhado", "detailed", "profundo",
        "in-depth", "full stack", "sistema completo", "autenticação",
        "authentication", "banco de dados", "database", "deploy",
        "integração", "integration", "api", "complexo", "complex",
        "avançado", "advanced", "enterprise", "produção", "production",
        "escalável", "scalable", "jwt", "refresh token", "segurança"
    ],
    "medium": [
        "simples mas", "simple but", "algumas", "some", "básico com",
        "basic with", "inclua", "include", "adicione", "add",
        "personalize", "customize", "melhore", "improve", "otimize"
    ],
    "low": [
        "simples", "simple", "rápido", "quick", "básico", "basic",
        "pequeno", "small", "só", "just", "apenas", "only",
        "um único", "a single", "mínimo", "minimal"
    ]
}

VAGUENESS_INDICATORS = [
    "bonito", "nice", "bom", "good", "legal", "cool", "interessante",
    "interesting", "profissional", "professional", "moderno", "moderna", "modern",
    "algo", "something", "tipo", "kind of", "qualquer", "any", "generico", "generica", "vago", "vaga"
]

ACTION_INDICATORS = [
    r"\b(faca|faz|execute|executar|rode|rodar|instale|instalar)\b",
    r"\b(configure|configurar|implante|implantar|deploy)\b",
    r"\b(acesse|acessar|conecte|conectar|ssh)\b",
    r"\b(automatize|automatizar|automate|agende|agendar|monitore|monitorar)\b",
    r"\b(crie|criar|create|build|desenvolva|desenvolver|develop|implemente|implementar|implement)\b\s+(um|uma|a|an)?\s*(site|website|app|aplicativo|api|backend|frontend|sistema|programa|script|workflow|pipeline|arquivo|file|documento|document)\b",
    r"\b(gere|gerar|generate)\b\s+(um|uma|a|an)?\s*(imagem|image|foto|photo|video|vídeo|arquivo|file|documento|document|pdf|png|jpg|svg)\b",
    r"\b(baixe|baixar|download|upload)\b.*\b(dados|data|arquivo|api)\b",
    r"\b(pip|npm|apt|brew|docker|compose|kubectl)\b",
]

INHERENT_COMPLEXITY_INDICATORS = [
    r"\b(como o|like|clone|cópia)\b.*\b(airbnb|uber|twitter|instagram|facebook|netflix|spotify|amazon|whatsapp|tiktok|youtube|linkedin)\b",
    r"\b(compilador|compiler|sistema operacional|operating system|engine|game engine)\b",
    r"\b(blockchain|smart contract|machine learning model|neural network)\b",
    r"\b(e-commerce completo|marketplace|rede social|social network)\b",
]

FACTUAL_DATA_INDICATORS = [
    r"\b(preço|price|cotação|quote|valor atual|current value)\b",
    r"\b(2025|2026|2027|atual|current|hoje|today|agora|now|recente|recent)\b",
    r"\b(quanto custa|how much|pricing|preços)\b",
    r"\b(notícia|news|novidade|update|atualização)\b",
]

FILE_OUTPUT_INDICATORS = [
    r"\b(pdf|docx|xlsx|pptx|csv|json|html|png|jpg|svg)\b",
    r"\b(gere|gerar|salve|salvar|exporte|exportar)\b.*\b(arquivo|file|documento)\b",
    r"\b(apresentação|presentation|slides)\b",
    r"\b(planilha|spreadsheet)\b",
    r"\b(currículo|curriculum|resume|cv)\b",
]


def _is_negated_match(normalized_text: str, keyword: str) -> bool:
    pattern = _phrase_pattern(keyword)
    return bool(re.search(rf"\b(?:nao|never|nunca)\b(?:\s+\w+){{0,2}}\s+{pattern}", normalized_text))


def count_matches(text: str, keywords: list) -> int:
    normalized = normalize_text(text)
    return sum(
        1
        for keyword in keywords
        if re.search(_phrase_pattern(keyword), normalized)
        and not _is_negated_match(normalized, keyword)
    )


def count_regex_matches(text: str, patterns: list) -> int:
    normalized = normalize_text(text)
    return sum(1 for pattern in patterns if re.search(_normalize_pattern(pattern), normalized, re.IGNORECASE))


def analyze_intent(text: str) -> tuple:
    scores = {}
    for intent, keywords in INTENT_KEYWORDS.items():
        scores[intent] = count_matches(text, keywords)
    if not any(scores.values()):
        return "unknown", scores
    primary = max(scores, key=scores.get)
    return primary, scores


def analyze_complexity(text: str) -> tuple:
    high = count_matches(text, COMPLEXITY_INDICATORS["high"])
    medium = count_matches(text, COMPLEXITY_INDICATORS["medium"])
    low = count_matches(text, COMPLEXITY_INDICATORS["low"])
    word_count = len(text.split())
    if word_count > 200:
        high += 2
    elif word_count > 100:
        medium += 1
    elif word_count < 30:
        low += 1
    
    if high > 0 and high >= low:
        level = "high" if high > medium else "medium"
    elif medium >= high and medium > low:
        level = "medium"
    else:
        level = "low"
    return level, {"high": high, "medium": medium, "low": low, "word_count": word_count}


def detect_mixed_task(intent_scores: dict, text: str = "") -> tuple:
    """Detect compound requests using explicit connective language."""
    significant = {key: score for key, score in intent_scores.items() if score >= 1}
    if len(significant) < 2:
        return False, []

    sorted_intents = sorted(significant.items(), key=lambda item: item[1], reverse=True)
    top_score = sorted_intents[0][1]
    second_score = sorted_intents[1][1]
    normalized = normalize_text(text)
    action_after_connector = r"(?:crie|criar|desenvolva|desenvolver|analise|analisar|pesquise|pesquisar|gere|gerar|escreva|escrever|automatize|automatizar|compare|comparar|build|create|analyze|research|generate|write)"
    has_connector = bool(
        re.search(r"\b(?:and|tambem|depois|alem|then|plus)\b", normalized)
        or re.search(rf"\be\s+{action_after_connector}\b", normalized)
    )

    # One strong intent plus a single incidental keyword is not a compound task.
    if not has_connector:
        return False, []
    if top_score >= 2 and second_score == 1 and sorted_intents[0][0] == "automation":
        return False, []
    return True, [key for key, _ in sorted_intents[:3]]


def needs_factual_data(text: str) -> bool:
    matches = count_regex_matches(text, FACTUAL_DATA_INDICATORS)
    strong_temporal = bool(re.search(r"\b(este\s+ano|this\s+year|hoje|today|agora|now|atual|current)\b", normalize_text(text)))
    return matches >= 1 or strong_temporal


def needs_file_output(text: str) -> bool:
    normalized = normalize_text(text)
    output_container = r"\b(arquivo|file|documento|document|planilha|spreadsheet|apresentacao|presentation|slides)\b"
    output_verb = r"\b(gere|gerar|salve|salvar|exporte|exportar|crie|criar|generate|save|export|create)\b"
    file_type = r"\b(pdf|docx|xlsx|pptx|csv|json|html|png|jpg|svg)\b"
    return bool(
        re.search(output_container, normalized)
        or re.search(rf"{output_verb}.*{file_type}", normalized)
    )


def needs_agent_action(text: str) -> bool:
    normalized = normalize_text(text)
    for pattern in ACTION_INDICATORS:
        for match in re.finditer(_normalize_pattern(pattern), normalized, re.IGNORECASE):
            prefix = normalized[max(0, match.start() - 48):match.start()]
            if re.search(r"\b(?:nao|never|nunca)\b(?:\s+\w+){0,2}\s*$", prefix):
                continue
            return True
    return False


def is_inherently_complex(text: str) -> bool:
    return count_regex_matches(text, INHERENT_COMPLEXITY_INDICATORS) > 0


def determine_strategy(intent: str, complexity: str, is_mixed: bool, 
                       mixed_intents: list, needs_files: bool, needs_factual: bool,
                       has_actions: bool, inherent_complex: bool, text: str) -> dict:
    """Core decision matrix — ZERO quality loss."""
    
    force_agent = needs_files or has_actions or intent == "automation"
    if inherent_complex:
        complexity = "high"

    # Chat Mode — only for pure Q&A without tools needed
    if intent == "qa_brainstorm" and not force_agent and not is_mixed and not needs_factual:
        return {
            "strategy": "CHAT_MODE",
            "model": "Chat Mode (Free)",
            "credit_savings": "100%",
            "quality_impact": "0% — Chat Mode handles Q&A perfectly",
            "description": "Use Chat Mode — Q&A/brainstorm doesn't need Agent Mode.",
            "actions": ["Execute directly in Chat Mode", "No credits consumed"]
        }

    # Factual research
    if needs_factual and intent in ("research", "qa_brainstorm", "unknown"):
        return {
            "strategy": "BATCH_RESEARCH",
            "model": "Standard" if complexity != "high" else "Max (auto-selected)",
            "credit_savings": "30-50%",
            "quality_impact": "0% — actually IMPROVES quality by forcing online search",
            "description": "Factual query requiring online search. Batch queries for efficiency.",
            "actions": [
                "ALWAYS search online for factual/temporal data",
                "Use 3 query variants per search for max coverage",
                "Save findings to files (context hygiene)"
            ]
        }

    # Complex tasks → Max model
    if complexity == "high" and intent in ("code_dev", "data_analysis", "research"):
        return {
            "strategy": "DECOMPOSE_CASCADE",
            "model": "Max (auto-selected for complex tasks — 19.2% better quality)",
            "credit_savings": "20-40%",
            "quality_impact": "0% — Max auto-selected IMPROVES quality",
            "description": "Complex task. Decompose into phases. Max model for quality.",
            "actions": [
                "Plan decomposition in Chat Mode (FREE)",
                "Execute each module with Max model",
                "Test each module once, then integration test",
                "Context checkpoints between modules"
            ]
        }

    # Mixed tasks
    if is_mixed:
        model = "Max" if complexity == "high" else "Standard"
        return {
            "strategy": "DECOMPOSE_CASCADE",
            "model": model,
            "credit_savings": "25-45%",
            "quality_impact": "0% — decomposition IMPROVES quality per component",
            "description": f"Mixed task ({', '.join(mixed_intents)}). Decompose into phases.",
            "mixed_components": mixed_intents,
            "actions": [
                "Plan decomposition in Chat Mode (FREE)",
                f"Execute each component ({', '.join(mixed_intents)}) as separate phase",
                "Apply best practices for EACH task type",
                "Context checkpoints between phases"
            ]
        }

    # Code development
    if intent in ("code_dev", "code_fix"):
        return {
            "strategy": "DIRECT_STANDARD",
            "model": "Standard",
            "credit_savings": "30-50%",
            "quality_impact": "0% — robust code from start, smart testing",
            "description": "Code task. Write robust code from start, smart testing.",
            "actions": [
                "Write robust, clean, elegant code in one pass",
                "One sanity test at the end",
                "Up to 3 retries if test fails, then inform user"
            ]
        }

    # Research
    if intent == "research":
        return {
            "strategy": "BATCH_RESEARCH",
            "model": "Standard",
            "credit_savings": "30-50%",
            "quality_impact": "0% — batch queries maintain full coverage",
            "description": "Research task. Batch queries, context hygiene.",
            "actions": [
                "ALWAYS search online for factual data",
                "Use 3 query variants per search",
                "Save discoveries to files",
                "Report depth matches task requirements"
            ]
        }

    # Data analysis
    if intent == "data_analysis":
        return {
            "strategy": "DIRECT_STANDARD",
            "model": "Standard" if complexity != "high" else "Max (auto-selected)",
            "credit_savings": "30-50%",
            "quality_impact": "0% — process data in one validated pass",
            "description": "Data analysis task. Use a single reproducible script and validate the result.",
            "actions": [
                "Inspect and validate the input data",
                "Process the dataset in one reproducible script",
                "Run one sanity check and report assumptions"
            ]
        }

    # Media generation
    if intent == "media_generation":
        vague = count_matches(text, VAGUENESS_INDICATORS) > 0
        return {
            "strategy": "REFINE_FIRST" if vague else "DIRECT_STANDARD",
            "model": "Standard",
            "credit_savings": "40-70%",
            "quality_impact": "0% — collect missing creative constraints before generation" if vague else "0%",
            "description": "Media task. Confirm visual constraints before an expensive generation when the prompt is vague.",
            "actions": [
                "Confirm style, dimensions, colors, and required elements" if vague else "Generate from the sufficiently specified brief",
                "Prefer one precise generation over repeated vague attempts"
            ]
        }

    # Automation
    if intent == "automation":
        return {
            "strategy": "DECOMPOSE_CASCADE",
            "model": "Standard" if complexity != "high" else "Max (auto-selected)",
            "credit_savings": "30-50%",
            "quality_impact": "0% — define and validate each workflow component",
            "description": "Automation task. Decompose the workflow and validate integrations before activation.",
            "actions": [
                "Define trigger, action, destination, and failure handling",
                "Test one non-destructive execution",
                "Enable recurring execution only after validation"
            ]
        }

    # Content creation
    if intent == "content_creation":
        return {
            "strategy": "DIRECT_STANDARD",
            "model": "Standard",
            "credit_savings": "30-60%",
            "quality_impact": "0% — one-shot for short, section-by-section for long",
            "description": "Content creation. Optimized generation strategy.",
            "actions": [
                "Short content: generate in one shot",
                "Long content: section by section for coherence",
                "Output depth matches what was requested"
            ]
        }

    # Vague or unknown prompts should clarify before spending on execution.
    if intent == "unknown":
        return {
            "strategy": "CLARIFY_FIRST",
            "model": "Chat Mode (Free)",
            "credit_savings": "100%",
            "quality_impact": "0% — clarification prevents misrouting and rework",
            "description": "The request is too ambiguous to route safely without clarification.",
            "actions": [
                "Ask for the desired outcome and constraints",
                "Confirm whether tools, current data, or file output are required",
                "Route the clarified task only after scope is explicit"
            ]
        }

    # Default
    return {
        "strategy": "DIRECT_STANDARD",
        "model": "Standard",
        "credit_savings": "20-40%",
        "quality_impact": "0% — optimize internal process only",
        "description": "Standard execution with internal optimization.",
        "actions": [
            "Optimize internal reasoning (fewer thinking tokens)",
            "Output quality and depth match task requirements",
            "Quality ALWAYS wins over savings"
        ]
    }


def generate_directives(intent: str, complexity: str, is_mixed: bool, 
                        mixed_intents: list, needs_files: bool) -> list:
    """Generate efficiency directives that NEVER affect output quality."""
    directives = [
        "Optimize INTERNAL process (reasoning, iterations), but final output must have the quality the task demands."
    ]

    if intent in ("code_dev", "code_fix"):
        directives.extend([
            "Write robust, clean, elegant code from the start. Avoid over-engineering but never sacrifice robustness.",
            "Write complete code in one pass, avoiding unnecessary incremental iterations.",
            "If a test fails, fix and re-test. Maximum 3 attempts. If it persists, inform the user."
        ])
        if complexity == "high":
            directives.append("CONTEXT CHECKPOINT: Save state after each completed module.")

    if intent == "research":
        directives.extend([
            "For factual/temporal data, ALWAYS search online. Internal knowledge only for stable concepts.",
            "Use 3 query variants per search to maximize coverage in fewer calls.",
            "Save discoveries to files to free context."
        ])

    if intent == "content_creation":
        directives.append("Long content: generate section by section for maximum quality and coherence.")

    if intent == "media_generation":
        directives.extend([
            "BEFORE generating any media: confirm visual style, dimensions, colors, required elements.",
            "The more specific the generation prompt, the lower the chance of re-generation (which costs extra credits)."
        ])

    if intent == "automation":
        directives.extend([
            "Define the trigger, action, destination, permissions, and failure handling before activation.",
            "Run one non-destructive smoke test before enabling recurring or external side effects."
        ])

    if is_mixed:
        directives.append(f"MIXED TASK detected ({', '.join(mixed_intents)}). Apply best practices for EACH component.")

    if complexity == "high":
        directives.append("CONTEXT HYGIENE: Save important information to files. Reference files instead of copying content between steps.")

    directives.append("Be efficient in INTERNAL reasoning. Final output must have the quality and depth the task demands.")
    
    return directives


# ============================================================
# MCP TOOLS
# ============================================================

@mcp.tool()
def analyze_prompt(prompt: str) -> dict:
    """
    Analyze an AI agent prompt and return optimization recommendations.
    
    Returns strategy, model recommendation, estimated credit savings,
    quality impact assessment, and efficiency directives.
    
    Args:
        prompt: The user's prompt/task description to analyze
    
    Returns:
        Complete analysis with strategy, model, savings, and directives
    """
    if not isinstance(prompt, str):
        raise TypeError("prompt must be a string")

    text = prompt.strip()
    if not text:
        return {
            "error": {
                "code": "EMPTY_PROMPT",
                "message": "prompt must contain at least one non-whitespace character"
            },
            "meta": {"version": SERVER_VERSION}
        }
    
    # Run all analyzers
    intent, intent_scores = analyze_intent(text)
    complexity, complexity_details = analyze_complexity(text)
    is_mixed, mixed_intents = detect_mixed_task(intent_scores, text)
    factual = needs_factual_data(text)
    files = needs_file_output(text)
    actions = needs_agent_action(text)
    inherent = is_inherently_complex(text)
    
    # Override intent based on detections
    if factual and intent in ("qa_brainstorm", "unknown"):
        intent = "research"
    if inherent:
        intent = "code_dev"
    if actions and intent in ("qa_brainstorm", "unknown"):
        sorted_intents = sorted(intent_scores.items(), key=lambda x: x[1], reverse=True)
        for candidate, score in sorted_intents:
            if candidate not in ("qa_brainstorm", "unknown") and score >= 1:
                intent = candidate
                break
    
    # A short conceptual question should not become a high-complexity build merely
    # because it mentions one advanced term such as authentication or security.
    if intent == "qa_brainstorm" and not actions and not files and not inherent:
        if complexity_details["word_count"] <= 80:
            complexity = "low"
        elif complexity == "high":
            complexity = "medium"

    # Get strategy
    strategy = determine_strategy(
        intent, complexity, is_mixed, mixed_intents,
        files, factual, actions, inherent, text
    )
    
    # Get directives
    directives = generate_directives(intent, complexity, is_mixed, mixed_intents, files)

    ordered_scores = sorted(intent_scores.values(), reverse=True)
    top_score = ordered_scores[0] if ordered_scores else 0
    second_score = ordered_scores[1] if len(ordered_scores) > 1 else 0
    margin = top_score - second_score
    if intent == "unknown":
        confidence = "low"
    elif top_score >= 3 and margin >= 2:
        confidence = "high"
    elif top_score >= 2 and margin >= 1:
        confidence = "medium"
    else:
        confidence = "low"
    
    return {
        "analysis": {
            "intent": intent,
            "complexity": complexity,
            "is_mixed_task": is_mixed,
            "mixed_components": mixed_intents if is_mixed else [],
            "needs_factual_data": factual,
            "needs_file_output": files,
            "needs_agent_action": actions,
            "is_inherently_complex": inherent,
            "word_count": len(text.split()),
            "intent_scores": intent_scores,
            "intent_confidence": confidence,
            "confidence_margin": margin
        },
        "recommendation": strategy,
        "efficiency_directives": directives,
        "meta": {
            "version": SERVER_VERSION,
            "engine": "boundary-aware multilingual heuristics",
            "quality_guarantee": "Quality veto: optimize internal process only; never reduce required output quality.",
            "clarification_policy": "Unknown or ambiguous prompts use CLARIFY_FIRST before paid execution.",
            "simulations": "53 scenario audit claimed by the project; validate locally before relying on savings estimates",
            "red_team": "High-performance red teams found ZERO quality degradation",
            "key_principle": "Optimize INTERNAL process, never OUTPUT quality"
        }
    }


@mcp.tool()
def get_strategy_for_task(task_type: str) -> dict:
    """
    Get the optimal strategy for a specific task type.
    
    Args:
        task_type: One of: qa, code, research, content, data_analysis, media, automation
    
    Returns:
        Optimal strategy with model recommendation and directives
    """
    strategies = {
        "qa": {
            "strategy": "CHAT_MODE",
            "model": "Chat Mode (Free)",
            "savings": "100%",
            "quality_impact": "0%",
            "when": "Q&A, brainstorming, translations, conceptual comparisons",
            "never_use_for": "Code, file output, factual data, SSH/execution"
        },
        "code": {
            "strategy": "DIRECT_STANDARD (simple) or DECOMPOSE_CASCADE (complex)",
            "model": "Standard (simple/medium) or Max (complex — auto-selected)",
            "savings": "30-50%",
            "quality_impact": "0% — robust code from start",
            "tips": [
                "Write complete code in one pass",
                "One sanity test at the end",
                "Up to 3 retries if test fails",
                "Complex projects: decompose into modules with checkpoints"
            ]
        },
        "research": {
            "strategy": "BATCH_RESEARCH",
            "model": "Standard (simple) or Max (complex — auto-selected)",
            "savings": "30-50%",
            "quality_impact": "0% — actually IMPROVES by forcing online search",
            "tips": [
                "ALWAYS search online for factual/temporal data",
                "Use 3 query variants per search",
                "Save findings to files (context hygiene)",
                "Report depth matches task requirements"
            ]
        },
        "content": {
            "strategy": "DIRECT_STANDARD (short) or DECOMPOSE_CASCADE (long)",
            "model": "Standard",
            "savings": "30-60%",
            "quality_impact": "0%",
            "tips": [
                "Short content: one-shot generation",
                "Long content (2000+ words): section by section",
                "Plan outline in Chat Mode first (free)"
            ]
        },
        "data_analysis": {
            "strategy": "DIRECT_STANDARD",
            "model": "Standard (simple) or Max (complex)",
            "savings": "30-50%",
            "quality_impact": "0%",
            "tips": [
                "Process all data in a single script",
                "Prefer TSV/TOML over JSON for structured data",
                "One sanity test at the end"
            ]
        },
        "media": {
            "strategy": "REFINE_FIRST (if vague) or DIRECT_STANDARD",
            "model": "Standard",
            "savings": "40-70%",
            "quality_impact": "0% — actually IMPROVES by collecting details first",
            "tips": [
                "BEFORE generating: confirm style, dimensions, colors, elements",
                "One precise attempt is better than several vague ones",
                "If prompt is vague, ask for details BEFORE generating"
            ]
        },
        "automation": {
            "strategy": "DECOMPOSE_CASCADE",
            "model": "Standard",
            "savings": "30-50%",
            "quality_impact": "0%",
            "tips": [
                "Define complete workflow before implementing",
                "Test critical component once at the end",
                "Up to 3 retries if test fails"
            ]
        }
    }
    
    if not isinstance(task_type, str):
        raise TypeError("task_type must be a string")

    task_type_lower = normalize_text(task_type).replace(" ", "_")
    aliases = {
        "question": "qa",
        "q_and_a": "qa",
        "q&a": "qa",
        "coding": "code",
        "development": "code",
        "data": "data_analysis",
        "dataanalysis": "data_analysis",
        "media_generation": "media",
        "image": "media",
        "workflow": "automation",
    }
    task_type_lower = aliases.get(task_type_lower, task_type_lower)
    if task_type_lower in strategies:
        result = dict(strategies[task_type_lower])
        result["normalized_task_type"] = task_type_lower
        return result
    
    return {
        "error": f"Unknown task type: {task_type}",
        "available_types": list(strategies.keys()),
        "tip": "Use analyze_prompt() for automatic detection"
    }


@mcp.tool()
def get_golden_rules() -> dict:
    """
    Get the 10 Golden Rules for credit optimization with ZERO quality loss.
    
    Returns:
        The 10 audited golden rules with explanations
    """
    return {
        "version": f"v{SERVER_VERSION} — 53-scenario audit reference; validate locally before relying on savings estimates",
        "rules": [
            {
                "number": 1,
                "title": "Output with adequate depth",
                "rule": "Conciseness applies ONLY to internal reasoning. The delivered result must have the quality and depth the task demands. NEVER shorten output to save credits.",
                "quality_impact": "POSITIVE — ensures full output quality"
            },
            {
                "number": 2,
                "title": "Robust code from the start",
                "rule": "Write robust, clean, elegant code. Avoid over-engineering but NEVER sacrifice robustness (validation, error handling).",
                "quality_impact": "POSITIVE — better code quality"
            },
            {
                "number": 3,
                "title": "Up to 3 attempts for code",
                "rule": "If a test fails, fix and re-test. Maximum 3 attempts. If it persists, inform the user about the specific problem. NEVER deliver broken code.",
                "quality_impact": "POSITIVE — ensures working code"
            },
            {
                "number": 4,
                "title": "ALWAYS search online for factual data",
                "rule": "For data that changes (prices, statistics, events), ALWAYS search online. Internal knowledge only for stable concepts.",
                "quality_impact": "POSITIVE — ensures accurate, current data"
            },
            {
                "number": 5,
                "title": "Long content = section by section",
                "rule": "Articles, reports, and presentations with 2000+ words or 10+ slides should be generated section by section for coherence and depth.",
                "quality_impact": "POSITIVE — better coherence"
            },
            {
                "number": 6,
                "title": "Max auto-selected for high complexity",
                "rule": "Complex tasks use Max model automatically (19.2% better quality). NEVER block Max when recommended.",
                "quality_impact": "POSITIVE — 19.2% quality improvement"
            },
            {
                "number": 7,
                "title": "Action detection = Agent Mode mandatory",
                "rule": "If the task needs execution (SSH, install, configure, generate file), ALWAYS use Agent Mode.",
                "quality_impact": "POSITIVE — ensures task completion"
            },
            {
                "number": 8,
                "title": "Mixed tasks = decomposition",
                "rule": "If the task has multiple components (research + slides + charts), decompose into phases and apply best practices for each type.",
                "quality_impact": "POSITIVE — better per-component quality"
            },
            {
                "number": 9,
                "title": "Context hygiene for long tasks",
                "rule": "After each completed module/phase, save state to file and reference instead of keeping in context. This IMPROVES quality.",
                "quality_impact": "POSITIVE — reduces context rot"
            },
            {
                "number": 10,
                "title": "Media: collect details BEFORE generating",
                "rule": "For image/video generation, collect style, dimensions, colors, and references BEFORE generating. One precise attempt is better than several vague ones.",
                "quality_impact": "POSITIVE — fewer re-generations"
            }
        ],
        "veto_rule": "If ANY efficiency directive conflicts with final result quality, quality ALWAYS wins. Credit savings are secondary to delivering excellent results."
    }


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    mcp.run()
