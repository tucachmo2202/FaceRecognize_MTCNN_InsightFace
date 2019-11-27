import os
data_dir = "mydata/"
persons = os.listdir(data_dir)
file = open("sorted-train.txt", "w")
for person in persons:
    imgs = os.listdir(data_dir + person + "/")
    for img in imgs:
        file.write(person+ "\t" + person + "/" + img + "\n")
file.close()