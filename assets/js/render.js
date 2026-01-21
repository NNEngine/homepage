function getFileFromURL() {
    const params = new URLSearchParams(window.location.search);
    return params.get("file");
}

async function loadMarkdown() {
    const file = getFileFromURL();
    if (!file) return;

    const response = await fetch(`content/${file}`);
    const text = await response.text();

    const html = marked.parse(text);
    const container = document.getElementById("content");

    container.innerHTML = html;

    // Render KaTeX math
    if (window.renderMathInElement) {
        renderMathInElement(container, {
            delimiters: [
                { left: "$$", right: "$$", display: true },
                { left: "$", right: "$", display: false },

                 // 🔥 ADD THESE
                { left: "\\(", right: "\\)", display: false },
                { left: "\\[", right: "\\]", display: true }
            ],
            throwOnError: false
        });
    }
}

function goHome() {
    window.location.href = "index.html";
}

loadMarkdown();
