// assets/app.js
(function () {
  const year = document.getElementById("year");
  const githubLink = document.getElementById("githubLink");

  if (year) year.textContent = new Date().getFullYear();

  // Global GitHub repo link (top-level)
  if (githubLink) githubLink.href = "https://github.com/Mech123/dswp_group2/tree/main";

  // Sidebar toggle
  const menuBtn = document.querySelector(".topbar .icon-btn");

  function setSidebarCollapsed(collapsed) {
    document.body.classList.toggle("sidebar-collapsed", collapsed);
    try {
      localStorage.setItem("sidebar-collapsed", collapsed ? "1" : "0");
    } catch (e) {}
  }

  // Restore state - default to collapsed (closed)
  try {
    const saved = localStorage.getItem("sidebar-collapsed");
    if (saved === null) {
      // No saved state, default to collapsed
      setSidebarCollapsed(true);
    } else if (saved === "1") {
      setSidebarCollapsed(true);
    }
  } catch (e) {
    // If error, default to collapsed
    setSidebarCollapsed(true);
  }

  if (menuBtn) {
    menuBtn.addEventListener("click", () => {
      const collapsed = document.body.classList.contains("sidebar-collapsed");
      setSidebarCollapsed(!collapsed);
    });
  }

  // Active nav link (based on current filename)
  const currentPath = (window.location.pathname || "").split("/").pop();
  document.querySelectorAll(".nav .nav-link").forEach((a) => {
    const href = (a.getAttribute("href") || "").split("/").pop();
    if (href && currentPath && href === currentPath) a.classList.add("active");
  });

  // ---------- Lightbox / Image Modal ----------
  function ensureImageModal() {
    if (document.getElementById("imgModal")) return;

    const modal = document.createElement("div");
    modal.id = "imgModal";
    modal.style.cssText = `
      position: fixed; inset: 0; display: none; z-index: 9999;
      background: rgba(0,0,0,0.75); padding: 24px;
      align-items: center; justify-content: center;
    `;

    modal.innerHTML = `
      <div style="position:relative; max-width: 1200px; width: 100%;">
        <button id="imgModalClose" aria-label="Close"
          style="
            position:absolute; top:-10px; right:-10px; width:40px; height:40px;
            border:none; border-radius: 999px; cursor:pointer;
            background:#fff; box-shadow: 0 8px 20px rgba(0,0,0,0.25);
          ">
          <span style="font-size:18px;">✕</span>
        </button>
        <div style="background:#fff; border-radius: 14px; overflow:hidden;">
          <img id="imgModalImg" alt="Preview"
            style="display:block; width:100%; height:auto; max-height: 80vh; object-fit: contain; background:#0b1220;">
          <div id="imgModalCaption" style="padding:12px 14px; color:#223;"></div>
        </div>
      </div>
    `;

    document.body.appendChild(modal);

    const closeBtn = document.getElementById("imgModalClose");
    const close = () => (modal.style.display = "none");

    closeBtn.addEventListener("click", close);
    modal.addEventListener("click", (e) => {
      if (e.target === modal) close();
    });

    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape") close();
    });
  }

  function openImageModal(src, caption) {
    ensureImageModal();
    const modal = document.getElementById("imgModal");
    const img = document.getElementById("imgModalImg");
    const cap = document.getElementById("imgModalCaption");

    img.src = src;
    cap.textContent = caption || "";
    modal.style.display = "flex";
  }

  // Attach modal to any image with data-modal="true"
  document.querySelectorAll("img[data-modal='true']").forEach((img) => {
    img.style.cursor = "zoom-in";
    img.addEventListener("click", () => {
      openImageModal(img.src, img.getAttribute("data-caption") || img.alt || "");
    });
  });
})();

