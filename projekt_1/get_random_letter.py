from random import randint

def get_random_letter():
    data_path = "D:\\data\\project_1\\data.csv"

    with open(data_path, "r") as f:
        lines = [row for row in f]

    i = randint(10000, 20000)
    return lines[i]