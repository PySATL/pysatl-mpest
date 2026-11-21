// Runs in <head>, before the first paint: applies the saved navigation width and table-of-contents
// state to <html>, so a page does not render with the defaults and then jump. navigation.js reads the
// same state later and only adds the controls.
(() => {
  // Pages whose table of contents starts collapsed until the reader opens it.
  const tocCollapsedPages = ["design/index"];

  const read = (key) => {
    try {
      return localStorage.getItem(key);
    } catch {
      return null;
    }
  };

  const root = document.documentElement;
  const width = Number(read("mpest.navigation-width"));
  if (width) root.style.setProperty("--navigation-width", Math.min(360, Math.max(180, width)) + "px");

  // Sphinx declares DOCUMENTATION_OPTIONS with const, so it is not a property of window.
  const pagename = typeof DOCUMENTATION_OPTIONS === "undefined" ? undefined : DOCUMENTATION_OPTIONS.pagename;
  const saved = read("mpest.toc-expanded");
  const expanded = saved === null ? !tocCollapsedPages.includes(pagename) : saved === "true";
  root.classList.toggle("navigation-toc-collapsed", !expanded);
})();
