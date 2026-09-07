import re

with open('web_dashboard.py', encoding='utf-8') as f:
    content = f.read()

# 1. Remove the splashScreen div completely
content = re.sub(
    r'\s*<div id="splashScreen"[^>]*>.*?</div>',
    '',
    content,
    flags=re.DOTALL
)

# 2. Remove splash CSS
content = re.sub(r'\s*\.splash-active \{[^}]+\}', '', content)
content = re.sub(r'\s*#splashScreen \{[^}]+\}', '', content)
content = re.sub(r'\s*#splashLogo \{[^}]+\}', '', content)
content = re.sub(r'\s*@keyframes float3D \{.*?\}', '', content, flags=re.DOTALL)

# 3. Remove the addEventListener splash JS block
content = re.sub(
    r"window\.addEventListener\('load'.*?\}\);\s*\n",
    '',
    content,
    flags=re.DOTALL
)
# Also remove bare addEventListener (without window.)
content = re.sub(
    r"addEventListener\('load'.*?\}\);\s*\n",
    '',
    content,
    flags=re.DOTALL
)

# 4. Make nav logo visible (remove opacity-0)
content = content.replace(
    'opacity-0 transition-opacity duration-700',
    'transition-opacity duration-300'
)

with open('web_dashboard.py', 'w', encoding='utf-8') as f:
    f.write(content)

# Verify splash is gone
if 'splashScreen' in content:
    print("WARNING: splashScreen still present!")
else:
    print("SUCCESS: splash screen removed")
