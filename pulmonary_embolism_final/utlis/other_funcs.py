

def get_rank_recursive(value, distribution_list, index_add=0):

    current_len = len(distribution_list)
    if current_len < 2:
        return index_add + 1
    split_id = int(current_len / 2)
    if value > distribution_list[split_id]:
        return get_rank_recursive(value, distribution_list[split_id:], index_add=split_id + index_add)
    if value < distribution_list[split_id]:
        return get_rank_recursive(value, distribution_list[0: split_id], index_add=index_add)


def get_rank_count(value, distribution_list):
    smaller = 0
    for i in range(len(distribution_list)):
        if value > distribution_list[i]:
            smaller += 1
    return smaller
