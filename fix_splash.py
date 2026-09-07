import re

with open('web_dashboard.py', encoding='utf-8') as f:
    content = f.read()

# Replace the broken complex splash JS with a dead-simple one
# Find the window.addEventListener('load') block and replace
old_block_pattern = r"window\.addEventListener\('load',.*?\}\);"
new_block = (
    "window.addEventListener('load', function() {\n"
    "            setTimeout(function() {\n"
    "                var splash = document.getElementById('splashScreen');\n"
    "                if (splash) {\n"
    "                    splash.style.opacity = '0';\n"
    "                    splash.style.transition = 'opacity 0.6s ease';\n"
    "                    setTimeout(function() {\n"
    "                        if (splash.parentNode) splash.parentNode.removeChild(splash);\n"
    "                        var navLogo = document.getElementById('navLogo');\n"
    "                        if (navLogo) navLogo.style.opacity = '1';\n"
    "                    }, 700);\n"
    "                }\n"
    "            }, 1800);\n"
    "        });"
)

patched = re.sub(old_block_pattern, new_block, content, flags=re.DOTALL)

with open('web_dashboard.py', 'w', encoding='utf-8') as f:
    f.write(patched)

# Verify
if 'setTimeout(function()' in patched:
    print("SUCCESS: Splash fixed cleanly")
else:
    print("ERROR: Pattern not found")
