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
    sidebar.style.setProperty("--navigation-width", boundedWidth + "px");
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
    sidebar.classList.toggle("navigation-toc-collapsed", !expanded);
    button.setAttribute("aria-expanded", expanded);
    const label = expanded ? "Скрыть оглавление" : "Показать оглавление";
    button.setAttribute("aria-label", label);
    button.title = label;
    button.textContent = expanded ? "›" : "‹";
  };
  setExpanded(!document.querySelector(".bd-article > section#design"));
  button.addEventListener("click", () => setExpanded(button.getAttribute("aria-expanded") === "false"));
  sidebar.after(button);
}

const primarySidebar = document.getElementById("pst-primary-sidebar");
const secondarySidebar = document.getElementById("pst-secondary-sidebar");
if (primarySidebar) setupNavigationResize(primarySidebar);
if (secondarySidebar?.querySelector("nav.page-toc")) {
  setupTocToggle(secondarySidebar);
}
