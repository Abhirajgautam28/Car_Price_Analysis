text=open('app.py','r',encoding='utf-8').read()
print('total length', len(text))
print('try count', text.count('\ntry:\n'))
print('except count', text.count('\nexcept'))
for i,line in enumerate(text.splitlines(),start=1):
    if line.strip().startswith('try:'):
        print('try at',i)
    if line.strip().startswith('except'):
        print('except at',i,line.strip())
