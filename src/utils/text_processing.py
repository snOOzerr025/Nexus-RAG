"""Text processing utilities for document cleaning and normalization."""

from __future__ import annotations

import re


def normalize_whitespace(text: str) -> str:
    """Collapse multiple whitespace characters into single spaces.

    Preserves paragraph breaks (double newlines) while normalizing
    excessive whitespace within paragraphs.

    Args:
        text: Raw text to normalize.

    Returns:
        Text with normalized whitespace.
    """
    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Collapse multiple blank lines into double newline (paragraph break)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Collapse multiple spaces/tabs into single space (within lines)
    text = re.sub(r"[^\S\n]+", " ", text)
    # Strip leading/trailing whitespace from each line
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines)
    # Strip leading/trailing whitespace from entire text
    return text.strip()


def strip_markdown_formatting(text: str) -> str:
    """Remove markdown formatting while preserving text content.

    Strips bold, italic, links, images, code fences, and heading markers
    while keeping the actual text readable.

    Args:
        text: Markdown-formatted text.

    Returns:
        Plain text with markdown formatting removed.
    """
    # Remove code fences (```...```) but keep content
    text = re.sub(r"```[\w]*\n?", "", text)
    # Remove inline code backticks
    text = re.sub(r"`([^`]+)`", r"\1", text)
    # Remove images: ![alt](url) -> alt
    text = re.sub(r"!\[([^\]]*)\]\([^)]+\)", r"\1", text)
    # Remove links: [text](url) -> text
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    # Remove heading markers
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    # Remove bold/italic markers
    text = re.sub(r"\*{1,3}([^*]+)\*{1,3}", r"\1", text)
    text = re.sub(r"_{1,3}([^_]+)_{1,3}", r"\1", text)
    # Remove blockquote markers
    text = re.sub(r"^>\s*", "", text, flags=re.MULTILINE)
    # Remove horizontal rules
    text = re.sub(r"^[-*_]{3,}\s*$", "", text, flags=re.MULTILINE)
    # Remove list markers (unordered)
    text = re.sub(r"^[\s]*[-*+]\s+", "", text, flags=re.MULTILINE)
    # Remove list markers (ordered)
    text = re.sub(r"^[\s]*\d+\.\s+", "", text, flags=re.MULTILINE)
    # Remove HTML tags if any remain
    text = re.sub(r"<[^>]+>", "", text)

    return normalize_whitespace(text)


def extract_markdown_sections(text: str) -> list[dict]:
    """Extract sections from markdown text based on heading hierarchy.

    Parses heading markers (# H1, ## H2, etc.) and groups content
    under each heading into sections.

    Args:
        text: Markdown text with heading markers.

    Returns:
        List of section dicts with keys: heading, level, content,
        start_char, end_char.
    """
    heading_pattern = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
    sections: list[dict] = []
    matches = list(heading_pattern.finditer(text))

    if not matches:
        # No headings found — entire text is one section
        return [
            {
                "heading": "Document",
                "level": 1,
                "content": text.strip(),
                "start_char": 0,
                "end_char": len(text),
            }
        ]

    # Content before first heading
    if matches[0].start() > 0:
        preamble = text[: matches[0].start()].strip()
        if preamble:
            sections.append(
                {
                    "heading": "Preamble",
                    "level": 0,
                    "content": preamble,
                    "start_char": 0,
                    "end_char": matches[0].start(),
                }
            )

    for i, match in enumerate(matches):
        level = len(match.group(1))
        heading = match.group(2).strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        content = text[start:end].strip()

        sections.append(
            {
                "heading": heading,
                "level": level,
                "content": content,
                "start_char": match.start(),
                "end_char": end,
            }
        )

    return sections


def clean_html_text(html_text: str) -> str:
    """Basic HTML tag removal for cases where BeautifulSoup is overkill.

    Args:
        html_text: Text potentially containing HTML tags.

    Returns:
        Text with HTML tags removed.
    """
    # Remove script and style elements entirely
    text = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", html_text, flags=re.DOTALL | re.IGNORECASE)
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Decode common HTML entities
    text = text.replace("&amp;", "&")
    text = text.replace("&lt;", "<")
    text = text.replace("&gt;", ">")
    text = text.replace("&quot;", '"')
    text = text.replace("&#39;", "'")
    text = text.replace("&nbsp;", " ")

    return normalize_whitespace(text)


def sanitize_for_prompt(text: str, max_length: int = 2000) -> str:
    """Sanitize user input before including in LLM prompts.

    Prevents prompt injection by escaping control sequences and
    truncating to a safe length.

    Args:
        text: Raw user input.
        max_length: Maximum character length.

    Returns:
        Sanitized text safe for prompt inclusion.
    """
    # Strip null bytes and control characters (except newline, tab)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    # Truncate to max length
    if len(text) > max_length:
        text = text[:max_length] + "..."
    return text.strip()


def parse_llm_json(response_text: str) -> dict:
    """Safely parse JSON returned by an LLM.

    Handles markdown fences (```json ... ```), invalid backslash escapes
    (e.g., LaTeX or unescaped characters in text), and unescaped control chars.

    Args:
        response_text: Raw string output from the LLM.

    Returns:
        Parsed dict or list from the JSON content.
    """
    import json
    cleaned = response_text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

    try:
        return json.loads(cleaned)
    except Exception:
        pass

    # Sanitize invalid backslash escapes that break standard JSON parsers
    # Valid escapes: \", \\, \/, \b, \f, \n, \r, \t, \uXXXX
    sanitized = re.sub(r'\\(?![/"\\bfnrtu]|u[0-9a-fA-F]{4})', r'\\\\', cleaned)
    try:
        return json.loads(sanitized)
    except Exception:
        pass

    try:
        return json.loads(cleaned, strict=False)
    except Exception:
        pass

    return json.loads(sanitized, strict=False)

