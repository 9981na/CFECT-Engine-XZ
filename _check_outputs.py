"""Check old outputs/ directory for data completeness"""
import os, json

# 1. First pass: directory structure
print("="*80)
print("OUTPUTS DIRECTORY STRUCTURE")
print("="*80)
for root, dirs, files in os.walk(r'E:\MEM\paper\real\outputs'):
    level = root.replace(r'E:\MEM\paper\real\outputs', '').count(os.sep)
    indent = ' ' * 2 * level
    print(f'{indent}{os.path.basename(root)}/')
    subindent = ' ' * 2 * (level + 1)
    for f in files:
        fsize = os.path.getsize(os.path.join(root, f))
        print(f'{subindent}{f} ({fsize:,} bytes)')

print("\n" + "="*80)
print("RUN LOGS (scanning for subject counts / errors)")
print("="*80)
for root, dirs, files in os.walk(r'E:\MEM\paper\real\outputs'):
    for f in files:
        if f == 'run.log':
            fp = os.path.join(root, f)
            with open(fp, 'r', encoding='utf-8', errors='replace') as fh:
                content = fh.read()
            print(f'\n--- {fp} ---')
            # Look for key patterns
            for line in content.split('\n'):
                ll = line.lower().strip()
                if any(kw in ll for kw in ['subject', 'patient', 'file', 'error', 'warning', 
                                             'complete', 'total', 'skip', 'fail', 'missing',
                                             'sample', 'record', 'window', 'epoch']):
                    print(f'  {line.strip()[:200]}')

print("\n" + "="*80)
print("RESULTS.JSON (if any)")
print("="*80)
for root, dirs, files in os.walk(r'E:\MEM\paper\real\outputs'):
    for f in files:
        if f == 'results.json':
            fp = os.path.join(root, f)
            with open(fp, 'r', encoding='utf-8', errors='replace') as fh:
                try:
                    data = json.load(fh)
                    s = json.dumps(data, indent=2)
                    print(f'{fp}:\n{s[:1000]}\n...')
                except:
                    print(f'{fp}: (unparseable)')

print("\n" + "="*80)
print("CSV FILES IN outputs/")
print("="*80)
for root, dirs, files in os.walk(r'E:\MEM\paper\real\outputs'):
    for f in files:
        if f.endswith('.csv'):
            fp = os.path.join(root, f)
            # Read first 3 lines for column headers
            with open(fp, 'r', encoding='utf-8', errors='replace') as fh:
                first_lines = [next(fh) for _ in range(3)]
            fsize = os.path.getsize(fp)
            print(f'{fp}')
            print(f'  Size: {fsize:,} bytes | Headers: {first_lines[0].strip()[:200]}')
            if len(first_lines) > 1:
                print(f'  Row 1: {first_lines[1].strip()[:200]}')
            print()
