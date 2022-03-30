import random
 
data_path = "D:\\data\\project_1\\data.csv" #"C:\\Users\\monik\\OneDrive\\Pulpit\\AIB\\AIB\\data.csv"
with open(data_path, "r") as f:
    lines = [row for row in f]
    random.shuffle(lines)
to_wrt = []
for i in range(100000):
    to_wrt.append(lines[i])
to_write = ""    
 
with open("C:\\Users\\Szymon\\AIB\\projekt_1\\data_mini.csv", "w") as save:
    for i in range(100000):
        to_write += to_wrt[i]
    save.write(to_write)
 