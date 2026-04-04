import glob

for fp in glob.glob('scripts/**/*.py', recursive=True):
    with open(fp, 'r', encoding='utf-8') as f:
        content = f.read()
    if r'\"' in content:
        content = content.replace(r'\"', '"')
        with open(fp, 'w', encoding='utf-8') as f:
            f.write(content)
        print('Fixed ' + fp)
