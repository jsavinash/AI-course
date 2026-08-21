#!/usr/bin/env python3
"""Convert the generator-produced HTML README.md files into GitHub-flavored Markdown.

Why this exists
---------------
scripts/generate_readmes.py emits full standalone HTML documents as README.md.
GitHub sanitizes README.md heavily: it strips <style>, <script>, <link>, <meta>,
<button>, and most class/id attributes, and it does NOT execute JavaScript. As a
result the HTML READMEs lose all styling, the copy buttons, KaTeX auto-render and
Mermaid JS. The content still *mostly* shows, but:
  * <style>/<script>/<link>/<meta>/<title> are dead weight
  * <pre class="mermaid"> never renders (no JS) -> must become ```mermaid
  * inline <code> blocks should be fenced code blocks
  * tables should be GFM tables so they render

This converter produces clean Markdown that renders correctly on GitHub while
keeping the same information (titles, sections, math, code, diagrams, API tables,
related links).
"""

import html
import os
import re
import sys
from html.parser import HTMLParser
from pathlib import Path

BLOCK_TAGS = {
    "html", "head", "body", "main", "header", "footer", "section", "div",
    "p", "h1", "h2", "h3", "h4", "h5", "h6", "ul", "ol", "li", "table",
    "thead", "tbody", "tr", "pre", "blockquote", "br", "hr", "article", "nav",
}
SKIP_TAGS = {"script", "style", "button", "meta", "link", "title", "noscript"}
# Tags emitted verbatim into the Markdown so GitHub renders them natively.
RAW_TAGS = {"details", "summary"}


class ReadmeConverter(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.out = []
        self.skip_depth = 0
        self.pre_depth = 0
        self.pre_lang = None
        self.pre_text = None
        self.inline_code = False
        self.link_stack = []
        self.list_stack = []  # list of (tag, index)
        self.title = None
        self.first_heading = True

    # -- helpers --------------------------------------------------------------
    def _text(self):
        return "".join(self.out)

    def _ensure_block_break(self):
        # Make sure buffer ends with at least one blank line so block elements
        # are separated in the rendered Markdown.
        if not self.out:
            return
        joined = "".join(self.out)
        if joined.endswith("\n\n"):
            return
        if joined.endswith("\n"):
            self.out.append("\n")
        else:
            self.out.append("\n\n")

    def _write(self, s):
        if self.skip_depth > 0:
            return
        if self.link_stack:
            self.link_stack[-1][1] += s
            return
        self.out.append(s)

    # -- tag handlers ---------------------------------------------------------
    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag in SKIP_TAGS:
            if tag == "title":
                # capture title text for the H1
                self._title_capture = True
                self.skip_depth += 1
                return
            if tag in ("meta", "link"):
                # void elements: no closing tag, never increment skip_depth
                return
            self.skip_depth += 1
            return

        if tag in RAW_TAGS:
            if tag == "details":
                self._ensure_block_break()
                self.out.append("<details>\n")
            elif tag == "summary":
                self.out.append("<summary>")
            return

        if tag == "pre":
            self.pre_depth += 1
            self.pre_lang = self._lang_from_attrs(attrs)
            self.pre_text = ""
            return

        if tag == "br":
            self._write("\n")
            return
        if tag == "hr":
            self._ensure_block_break()
            self._write("---\n\n")
            return

        if tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            self._ensure_block_break()
            level = int(tag[1])
            self.out.append("#" * level + " ")
            return

        if tag == "p":
            self._ensure_block_break()
            return

        if tag in ("ul", "ol"):
            self._ensure_block_break()
            self.list_stack.append([tag, 0])
            return
        if tag == "li":
            self._ensure_block_break()
            indent = "  " * (len(self.list_stack) - 1)
            if self.list_stack and self.list_stack[-1][0] == "ol":
                self.list_stack[-1][1] += 1
                marker = f"{self.list_stack[-1][1]}. "
            else:
                marker = "- "
            self.out.append(f"{indent}{marker}")
            return

        if tag == "table":
            self._ensure_block_break()
            return

        if tag == "code":
            if self.pre_depth > 0:
                if not self.pre_lang:
                    self.pre_lang = self._lang_from_attrs(attrs) or self.pre_lang
                return
            self.inline_code = True
            self._write("`")
            return

        if tag in ("strong", "b"):
            self._write("**")
            return
        if tag in ("em", "i"):
            self._write("*")
            return

        if tag == "a":
            href = attrs.get("href", "")
            self.link_stack.append([href, ""])
            return

        if tag == "img":
            alt = attrs.get("alt", "")
            src = attrs.get("src", "")
            self._write(f"![{alt}]({src})")
            return

        if tag == "blockquote":
            self._ensure_block_break()
            return

        # block wrapper tags (section/div/main/etc.) -> ensure separation only
        if tag in BLOCK_TAGS:
            self._ensure_block_break()

    def handle_endtag(self, tag):
        if tag in SKIP_TAGS:
            self.skip_depth = max(0, self.skip_depth - 1)
            if tag == "title":
                self._title_capture = False
            return

        if tag in RAW_TAGS:
            if tag == "details":
                self.out.append("</details>\n\n")
            elif tag == "summary":
                self.out.append("</summary>\n")
            return

        if tag == "pre":
            self.pre_depth -= 1
            text = self.pre_text or ""
            text = text.strip("\n")
            lang = self.pre_lang or ""
            self._ensure_block_break()
            fence = "```"
            self.out.append(f"{fence}{lang}\n")
            self.out.append(text + "\n")
            self.out.append(f"{fence}\n\n")
            self.pre_lang = None
            self.pre_text = None
            return

        if tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            self.out.append("\n\n")
            return

        if tag == "p":
            self.out.append("\n\n")
            return

        if tag in ("ul", "ol"):
            if self.list_stack:
                self.list_stack.pop()
            self._ensure_block_break()
            return
        if tag == "li":
            self.out.append("\n")
            return

        if tag == "code":
            if self.pre_depth > 0:
                return
            self.inline_code = False
            self._write("`")
            return

        if tag in ("strong", "b"):
            self._write("**")
            return
        if tag in ("em", "i"):
            self._write("*")
            return

        if tag == "a":
            if self.link_stack:
                href, txt = self.link_stack.pop()
                self._write(f"[{txt}]({href})")
            return

        if tag == "tr":
            self.out.append("\n")
            return

        if tag in BLOCK_TAGS:
            self._ensure_block_break()

    def handle_data(self, data):
        if self.skip_depth > 0 and not getattr(self, "_title_capture", False):
            if getattr(self, "_title_capture", False):
                self.title = (self.title or "") + data
            return
        if getattr(self, "_title_capture", False):
            self.title = (self.title or "") + data
            return
        if self.pre_depth > 0:
            self.pre_text += data
            return
        self._write(data)

    def handle_entityref(self, name):  # pragma: no cover - convert_charrefs handles
        self._write(f"&{name};")

    def handle_charref(self, name):  # pragma: no cover
        self._write(f"&#{name};")

    @staticmethod
    def _lang_from_attrs(attrs):
        classes = attrs.get("class", "")
        if "mermaid" in classes.split():
            return "mermaid"
        for c in classes.split():
            if c.startswith("language-"):
                return c[len("language-"):]
        return ""


def convert(html_text: str) -> str:
    # Pre-convert <table> blocks to GFM so the main parser can ignore them.
    html_text, table_map = _extract_tables(html_text)

    conv = ReadmeConverter()
    conv.feed(html_text)
    md = "".join(conv.out)

    # Restore tables at placeholders.
    for placeholder, gfm in table_map.items():
        md = md.replace(placeholder, gfm)

    # Prepend a title H1 only if the document has no H1 of its own
    # (DOCUMENTATION.html already carries a hero <h1>).
    title = conv.title
    if title and not md.lstrip().startswith("# "):
        title = title.strip()
        # title may use " - " or an em-dash "—"
        name = re.split(r"\s+[—-]\s+", title)[0].strip()
        md = f"# {name}\n\n" + md

    md = _cleanup(md)
    return md


def _extract_tables(html_text):
    table_map = {}
    idx = 0

    def repl(m):
        nonlocal idx
        placeholder = f"\n\n__TABLE_{idx}__\n\n"
        idx += 1
        table_map[placeholder] = _table_to_gfm(m.group(0))
        return placeholder

    return re.sub(r"<table.*?</table>", repl, html_text, flags=re.S), table_map


def _table_to_gfm(table_html):
    rows = re.findall(r"<tr.*?</tr>", table_html, flags=re.S)
    if not rows:
        return ""

    def cells(row_html):
        cells_html = re.findall(r"<t[hd].*?</t[hd]>", row_html, flags=re.S)
        out = []
        for c in cells_html:
            inner = re.sub(r"<t[hd][^>]*>", "", c)
            inner = re.sub(r"</t[hd]>", "", inner)
            # keep inline <code> as backticks
            inner = re.sub(r"<code[^>]*>(.*?)</code>", r"`\1`", inner, flags=re.S)
            inner = re.sub(r"<[^>]+>", "", inner)  # strip remaining tags
            inner = html.unescape(inner).strip().replace("\n", " ").replace("|", "\\|")
            out.append(inner)
        return out

    data_rows = [cells(r) for r in rows]
    header = data_rows[0]
    body = data_rows[1:]
    lines = []
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")
    for r in body:
        lines.append("| " + " | ".join(r) + " |")
    return "\n".join(lines) + "\n\n"


def _cleanup(md):
    # Collapse 3+ newlines to 2.
    md = re.sub(r"\n{3,}", "\n\n", md)
    # Remove trailing whitespace on lines.
    md = "\n".join(line.rstrip() for line in md.split("\n"))
    # Trim leading/trailing blank lines.
    md = md.strip() + "\n"
    return md


def main():
    if len(sys.argv) < 2:
        print("usage: html_readme_to_markdown.py <file.md> [out.md]", file=sys.stderr)
        sys.exit(1)
    src = Path(sys.argv[1])
    text = src.read_text(encoding="utf-8")
    md = convert(text)
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else src
    out.write_text(md, encoding="utf-8")
    print(f"Converted {src} -> {out} ({len(md)} bytes)")


if __name__ == "__main__":
    main()
