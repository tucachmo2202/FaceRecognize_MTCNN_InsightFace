def compute_matching(x):
    if x < 0.2:
        result= 99.6 + (0.2-x)/2
    elif (0.2 <= x) and (x < 0.4):
        result= 99 + (0.4 - x) * 2.5
    elif (0.4 <= x) and (x < 1.2):
        result= 91 + (1.2 - x) * 10
    elif (1.2 <= x) and (x < 1.41):
        result= 83 + (1.41 - x) * 38
    elif (1.41 <=x ) and (x < 1.71):
        result= 20 + (1.71-x)*200
    elif (1.71<=x):
        result= 10 + 1.71/x*10
    result = float("{0:.2f}".format(result))
    return result

def compute_distance(first_vector, second_vector):
    distance = 0
    for i in range(len(first_vector)):
        distance += (second_vector[i] - first_vector[i]) * (second_vector[i] - first_vector[i])
    return distance