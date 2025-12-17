/* ============================================================
   DIGITAL TWIN — CONTROLS.JS (Edition 8.1 Clean Version)
   Purpose: Only shooting stars + theme toggles
   Removed: starfield, milkyway, jwst-hex injection (handled by space.js)
   Safe for Dash — No ID access, No conflicts
============================================================ */

document.addEventListener("DOMContentLoaded", function () {

    console.log("Controls.js (Edition 8.1 Clean) Loaded Successfully 🚀");

    /* ---------------------------------------------------------
       1) Shooting stars (kept — looks good in cinematic mode)
    --------------------------------------------------------- */
    for (let i = 0; i < 5; i++) {
        let star = document.createElement("div");
        star.className = "shooting-star";
        star.style.left = Math.random() * window.innerWidth + "px";
        star.style.animationDelay = (Math.random() * 6) + "s";
        document.body.appendChild(star);
    }

    /* ---------------------------------------------------------
       2) Remove old parallax attempts (starfield/hex removed)
       Only cinema-space parallax from space.js is active
    --------------------------------------------------------- */

    /* ---------------------------------------------------------
       3) Theme toggles (future use, safe to keep)
    --------------------------------------------------------- */
    window.applyJWSTTheme = function () {
        document.body.classList.add("jwst-theme");
    };

    window.applyNASADefault = function () {
        document.body.classList.remove("jwst-theme");
    };

});
