import re

with open('web_dashboard.py', encoding='utf-8') as f:
    content = f.read()

new_js = (
    "addEventListener('load', () => {\n"
    "            const splash = document.getElementById('splashScreen');\n"
    "            const navLogo = document.getElementById('navLogo');\n"
    "            setTimeout(() => {\n"
    "                if (!splash) return;\n"
    "                splash.style.transition = 'opacity 0.8s ease';\n"
    "                splash.style.opacity = '0';\n"
    "                splash.style.pointerEvents = 'none';\n"
    "                if (navLogo) navLogo.style.opacity = '1';\n"
    "                setTimeout(() => { if(splash && splash.parentNode) splash.parentNode.removeChild(splash); }, 900);\n"
    "            }, 1500);\n"
    "        });"
)

patched = re.sub(
    r"addEventListener\('load',.*?(?=\n\s*const dropZone)",
    new_js + "\n        ",
    content,
    flags=re.DOTALL
)

with open('web_dashboard.py', 'w', encoding='utf-8') as f:
    f.write(patched)

print("Patched:", new_js[:60] in patched)
