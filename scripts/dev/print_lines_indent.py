lines=open('app.py','r',encoding='utf-8').read().splitlines()
for i,line in enumerate(lines[:240],start=1):
    print(f"{i:03d}: {len(line)-len(line.lstrip())} '{line}'")
