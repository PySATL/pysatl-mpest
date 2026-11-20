"""Embed a diagrams.net viewer next to links to ``.drawio`` files.

A link title names the sheets to show, separated by commas, for example
``[source](diagrams/reporting.drawio "Release, Draft")``: one viewer is
inserted per sheet, opened on that sheet. Without a title the viewer opens
on the first sheet.
"""

from __future__ import annotations

import json
import re
from html import escape
from pathlib import Path
from typing import TYPE_CHECKING, Any

from docutils import nodes  # type: ignore[import-untyped]
from sphinx import addnodes
from sphinx.util import logging

if TYPE_CHECKING:
    from collections.abc import Iterator

    from sphinx.application import Sphinx
    from sphinx.util.typing import ExtensionMetadata

VIEWER_SCRIPT_URL = "https://viewer.diagrams.net/js/viewer-static.min.js"
DRAWIO_SUFFIX = ".drawio"
SHEET_NAME = re.compile(r'<diagram\b[^>]*\bname="([^"]*)"')

logger = logging.getLogger(__name__)


def _iter_drawio_links(doctree: nodes.Element) -> Iterator[nodes.Element]:
    """Yield download references whose target is a ``.drawio`` file."""
    for node in doctree.findall(addnodes.download_reference):
        target = node.get("reftarget", "")
        if isinstance(target, str) and "://" not in target and target.endswith(DRAWIO_SUFFIX):
            yield node


def _viewer_node(xml: str, page: int) -> nodes.raw:
    """Build a raw HTML node with the diagrams.net viewer opened on ``page``."""
    config = {
        "highlight": "#000000",
        "nav": True,
        "auto-fit": True,
        "resize": False,
        # No zoom button: with it the viewer stops fitting the sheet on narrow screens.
        # The toolbar stays visible because touch screens have no hover to reveal the lightbox.
        "toolbar": "pages layers lightbox",
        "toolbar-nohide": True,
        "edit": None,
        "page": page,
        "xml": xml,
    }
    payload = escape(json.dumps(config, ensure_ascii=False), quote=True)
    html = (
        '<div class="mxgraph" '
        'style="max-width:100%;border:1px solid #e0e0e0;margin-bottom:1rem;" '
        f'data-mxgraph="{payload}">'
        "</div>"
    )
    return nodes.raw("", html, format="html")


def _on_doctree_resolved(app: Sphinx, doctree: nodes.document, docname: str) -> None:
    """Insert viewer blocks above their source links."""
    if app.builder.format != "html":
        return

    for link in list(_iter_drawio_links(doctree)):
        _, abspath = app.env.relfn2path(link["reftarget"], docname)
        path = Path(abspath)
        if not path.is_file():
            continue

        block = link.parent
        while block is not None and not isinstance(block, nodes.paragraph):
            block = block.parent
        if block is None or block.parent is None:
            continue

        xml = path.read_text(encoding="utf-8")
        sheets = SHEET_NAME.findall(xml)
        title = link.get("reftitle") or link.get("title", "")
        requested = [name.strip() for name in title.split(",") if name.strip()]
        pages: list[int] = []
        for name in requested:
            if name in sheets:
                pages.append(sheets.index(name))
            else:
                logger.warning("sheet %r not found in %s", name, link["reftarget"], location=link)
        if not requested:
            pages = [0]

        parent = block.parent
        if pages and "drawio-source" not in block["classes"]:
            block["classes"].append("drawio-source")
        for page in pages:
            parent.insert(parent.index(block), _viewer_node(xml, page))


def _on_html_page_context(
    app: Sphinx,
    pagename: str,
    templatename: str,
    context: dict[str, Any],
    doctree: nodes.document | None,
) -> None:
    """Load the viewer script only on pages that embed a diagram."""
    if doctree is None:
        return
    if next(_iter_drawio_links(doctree), None) is not None:
        app.add_js_file(VIEWER_SCRIPT_URL, loading_method="defer")


def setup(app: Sphinx) -> ExtensionMetadata:
    """Connect the extension to the Sphinx build."""
    app.connect("doctree-resolved", _on_doctree_resolved)
    app.connect("html-page-context", _on_html_page_context)
    app.add_css_file("drawio-viewer.css")
    return {"version": "1.0", "parallel_read_safe": True, "parallel_write_safe": True}
