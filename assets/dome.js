/* ============================================================
   Observatory Dome Opening Animation — Edition 7.5
   Activated once DOM is ready
============================================================ */

document.addEventListener("DOMContentLoaded", function () {

    // Create overlay container
    let dome = document.createElement("div");
    dome.id = "dome-overlay";

    // Left half
    let domeLeft = document.createElement("div");
    domeLeft.id = "dome-left";
    domeLeft.className = "dome-half";

    // Right half
    let domeRight = document.createElement("div");
    domeRight.id = "dome-right";
    domeRight.className = "dome-half";

    // Glow
    let glow = document.createElement("div");
    glow.id = "dome-glow";

    // Append
    dome.appendChild(domeLeft);
    dome.appendChild(domeRight);
    dome.appendChild(glow);
    document.body.appendChild(dome);

    console.log("Dome Opening Animation Initialized 🛰️");
});
