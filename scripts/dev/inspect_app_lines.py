with open('app.py','rb') as f:
    data=f.read().splitlines()
for i in range(200,220):
    print(i+1, data[i])
