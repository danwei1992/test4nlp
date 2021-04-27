
def insertion_sort(array):
    for i in range(len(array)):
        cur_index = i
        while array[cur_index-1] > array[cur_index] and cur_index-1 >= 0:
            array[cur_index], array[cur_index-1] = array[cur_index-1], array[cur_index]
            cur_index -= 1
    return array

array = [1,5,3,7,9]

result =  insertion_sort(array)

print(result)