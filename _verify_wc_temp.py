import re
from pathlib import Path

path = r'e:/MEM/paper/generated_manuscript/run8/_flagship/step5_draft/nature_draft_v10.md'
text = Path(path).read_text(encoding='utf-8')

# Check expansions
checks = {
    'Exp1 - clinical context': 'Epilepsy affects approximately 50 million',
    'Exp2 - rival exclusions': 'not merely a descriptive re-labelling',
    'Exp3 - causal emergence': 'provides independent evidence',
    'Exp4 - path integral': 'explore a continuum of possible paths',
}
for name, marker in checks.items():
    found = marker in text
    print(f'{name}: {"YES" if found else "MISSED"}')

# Word count
body = text.split('## References')[0]
content = body
content = re.sub(r'\$\$.*?\$\$', '', content, flags=re.DOTALL)
content = re.sub(r'\$.*?\$', '', content)
content = re.sub(r'\[.*?Figure.*?\]', '', content)
content = re.sub(r'\|.*?\|.*?\|.*?\|', '', content)
words = [w for w in content.split() if w.strip()]
print(f'\nTotal body text: {len(words)}')
