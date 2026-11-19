# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

sys.path.insert(0, os.path.abspath("../../"))

project = "Mpest"
copyright = "2025, Danil Totmyanin, Anton Kazancev"
author = "Danil Totmyanin, Anton Kazancev"
release = "1.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "myst_parser",
    "sphinx.ext.mathjax",
    "sphinxcontrib.mermaid",
    "sphinx.ext.autosummary",
]

autodoc_typehints = "description"
napoleon_use_param = True
napoleon_use_ivar = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_preprocess_types = False
napoleon_custom_sections = [
    ("Methods", "rubric"),  # или оставить пустым, если хочешь полностью свой RST
]

autosummary_generate = True
add_module_names = False
templates_path = ["_templates"]
exclude_patterns = []

# -- Myst Parser config --

myst_enable_extensions = [
    "dollarmath",
    "amsmath",
]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
mermaid_init_js = """
import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.esm.min.mjs'

const make_config = () => {
  let prefersDark = localStorage.getItem('theme') === 'dark' || (localStorage.getItem('theme') === null && window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches)
  return({
    startOnLoad:false,
    darkMode: prefersDark,
    theme: prefersDark ? "dark" : "default"
  })
}

const init_mermaid = () => {
    let graphs = document.querySelectorAll(".mermaid");
    [...graphs].forEach((element) => {
        if (!element.hasAttribute("data-source")) {
            element.setAttribute("data-source", element.innerText);
        }
        if (element.hasAttribute("data-processed")) {
            let new_elt = document.createElement("pre");
            let graph_source = element.getAttribute("data-source");
            new_elt.appendChild(document.createTextNode(graph_source));
            new_elt.classList.add("mermaid");
            new_elt.setAttribute("data-source", graph_source);
            element.replaceWith(new_elt);
        }
    });

    let config = make_config()
    mermaid.initialize(config);
    mermaid.run();
}

init_mermaid();

let theme_observer = new MutationObserver(init_mermaid);
let body = document.getElementsByTagName("body")[0];
theme_observer.observe(body, {attributes: true});
window.theme_observer = theme_observer;
"""

import os
import re

from sphinx.ext.autosummary import generate as autosummary


def autosummary_from_docstrings(app):
    """
    Ищет .. autosummary:: в докстрингах и генерит .rst-страницы.
    """
    # Паттерн для поиска блока autosummary
    pattern = re.compile(r"^\s*\.\. autosummary::.*?(?=^\S|\Z)", re.S | re.M)

    sources = []

    # Проходим по объектам из автодока
    for name, obj in list(app.env.domains["py"].data["objects"].items()):
        doc = app.env.doc2path(obj[0], base=None)
        if not os.path.isfile(doc):
            continue
        with open(doc, encoding="utf-8") as f:
            text = f.read()
        for match in pattern.finditer(text):
            sources.append(doc)

    if sources:
        autosummary.generate_autosummary_docs(
            sources, suffix=".rst", base_path=app.srcdir, imported_members=True, app=app
        )


def setup(app):
    app.connect("builder-inited", autosummary_from_docstrings)
