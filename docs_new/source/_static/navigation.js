// Pages whose table of contents starts collapsed until the reader opens it.
const TOC_COLLAPSED_PAGES = new Set(["design/index"]);
const WIDTH_KEY = "mpest.navigation-width";
const TOC_KEY = "mpest.toc-expanded";

function readSetting(key) {
  try {
    return localStorage.getItem(key);
  } catch {
    return null;
  }
}

function writeSetting(key, value) {
  try {
    localStorage.setItem(key, value);
  } catch {
    // Storage may be unavailable (private mode, blocked site data): keep the default.
  }
}

function setupNavigationResize(sidebar) {
  const minWidth = 180;
  const maxWidth = 360;
  const separator = document.createElement("div");
  separator.className = "navigation-resize d-print-none";
  separator.tabIndex = 0;
  separator.setAttribute("role", "separator");
  separator.setAttribute("aria-label", "Ширина меню навигации");
  separator.setAttribute("aria-orientation", "vertical");
  separator.setAttribute("aria-controls", sidebar.id);
  separator.setAttribute("aria-valuemin", minWidth);
  separator.setAttribute("aria-valuemax", maxWidth);
  sidebar.after(separator);

  const currentWidth = () => Math.round(sidebar.getBoundingClientRect().width);
  const announceWidth = () => {
    if (separator.offsetWidth) {
      separator.setAttribute("aria-valuenow", currentWidth());
    }
  };
  const resize = (width, save = true) => {
    const boundedWidth = Math.min(maxWidth, Math.max(minWidth, width));
    sidebar.style.setProperty("--navigation-width", boundedWidth + "px");
    if (save) writeSetting(WIDTH_KEY, boundedWidth);
    announceWidth();
  };

  const savedWidth = Number(readSetting(WIDTH_KEY));
  if (savedWidth) resize(savedWidth, false);

  let dragOffset = 0;
  separator.addEventListener("pointerdown", (event) => {
    if (!event.isPrimary || event.button !== 0) return;
    dragOffset = event.clientX - currentWidth();
    separator.setPointerCapture(event.pointerId);
    separator.blur();
    event.preventDefault();
  });
  separator.addEventListener("pointermove", (event) => {
    if (separator.hasPointerCapture(event.pointerId)) resize(event.clientX - dragOffset);
  });
  separator.addEventListener("keydown", (event) => {
    const widths = {
      ArrowLeft: currentWidth() - 16,
      ArrowRight: currentWidth() + 16,
      Home: minWidth,
      End: maxWidth,
    };
    if (!Object.hasOwn(widths, event.key)) return;
    event.preventDefault();
    resize(widths[event.key]);
  });
  window.addEventListener("resize", announceWidth);
  announceWidth();
}

function setupTocToggle(sidebar) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "btn btn-sm btn-outline-secondary navigation-toc-toggle d-print-none";
  button.setAttribute("aria-controls", sidebar.id);
  const setExpanded = (expanded) => {
    sidebar.classList.toggle("navigation-toc-collapsed", !expanded);
    button.setAttribute("aria-expanded", expanded);
    const label = expanded ? "Скрыть оглавление" : "Показать оглавление";
    button.setAttribute("aria-label", label);
    button.title = label;
    button.textContent = expanded ? "›" : "‹";
  };

  const saved = readSetting(TOC_KEY);
  // Sphinx declares DOCUMENTATION_OPTIONS with const, so it is not a property of window.
  const pagename = typeof DOCUMENTATION_OPTIONS === "undefined" ? undefined : DOCUMENTATION_OPTIONS.pagename;
  setExpanded(saved === null ? !TOC_COLLAPSED_PAGES.has(pagename) : saved === "true");
  button.addEventListener("click", () => {
    const expanded = button.getAttribute("aria-expanded") === "false";
    setExpanded(expanded);
    writeSetting(TOC_KEY, expanded);
  });
  // Placed right before the table of contents so the button stays next to it.
  sidebar.before(button);
}

const primarySidebar = document.getElementById("pst-primary-sidebar");
const secondarySidebar = document.getElementById("pst-secondary-sidebar");
if (primarySidebar) setupNavigationResize(primarySidebar);
if (secondarySidebar?.querySelector("nav.page-toc")) {
  setupTocToggle(secondarySidebar);
}
