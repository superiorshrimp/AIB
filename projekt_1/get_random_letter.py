from random import randint

def get_random_letter():
    data_path = "D:\\data\\project_1\\data.csv"

    with open(data_path, "r") as f:
        lines = [row for row in f]

    rand = randint(10000, 20000)
    lines = [lines[i].split(",") for i in range(len(lines))]
    ret = [[int(line[i]) for i in range(len(line))] for line in lines]
    return ret[rand]