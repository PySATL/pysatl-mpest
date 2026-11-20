// The saved state is applied to <html> before the first paint by navigation-early.js;
// this script adds the controls that change it.
const WIDTH_KEY = "mpest.navigation-width";
const TOC_KEY = "mpest.toc-expanded";

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
  const resize = (width) => {
    const boundedWidth = Math.min(maxWidth, Math.max(minWidth, width));
    document.documentElement.style.setProperty("--navigation-width", boundedWidth + "px");
    writeSetting(WIDTH_KEY, boundedWidth);
    announceWidth();
  };

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
    document.documentElement.classList.toggle("navigation-toc-collapsed", !expanded);
    button.setAttribute("aria-expanded", expanded);
    const label = expanded ? "Скрыть оглавление" : "Показать оглавление";
    button.setAttribute("aria-label", label);
    button.title = label;
    button.textContent = expanded ? "›" : "‹";
  };

  setExpanded(!document.documentElement.classList.contains("navigation-toc-collapsed"));
  button.addEventListener("click", () => {
    const expanded = button.getAttribute("aria-expanded") === "false";
    setExpanded(expanded);
    writeSetting(TOC_KEY, expanded);
  });
  // Fixed in the corner next to the table of contents, so adding it does not shift the layout.
  document.body.append(button);
}

const primarySidebar = document.getElementById("pst-primary-sidebar");
const secondarySidebar = document.getElementById("pst-secondary-sidebar");
if (primarySidebar) setupNavigationResize(primarySidebar);
if (secondarySidebar?.querySelector("nav.page-toc")) {
  setupTocToggle(secondarySidebar);
}
