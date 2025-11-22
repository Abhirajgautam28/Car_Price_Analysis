lines=open('app.py','r',encoding='utf-8').read().splitlines()
stack=[]
for i,line in enumerate(lines, start=1):
    s=line.lstrip()
    indent=len(line)-len(s)
    if s.startswith('try:'):
        stack.append((i,indent))
    elif s.startswith('except') or s.startswith('finally'):
        if not stack:
            print('Unmatched except/finally at',i)
        else:
            # find last try with same indent
            for j in range(len(stack)-1, -1, -1):
                if stack[j][1]==indent:
                    stack.pop(j)
                    break
            else:
                print('No try at same indent for except at',i,'indent',indent,'stack',stack[:5])

if stack:
    print('Unmatched try(s):')
    for t in stack:
        print(' try at',t)
else:
    print('All try/except/finally matched (by indent)')
