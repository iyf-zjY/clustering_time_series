import numpy as np
import csv


# generate a single line of stock graph
def ask_shape_similar(stock_no, mem, data_ud, stock_idx):
    if stock_no not in stock_idx:
        return [stock_no]
    this_type = mem[stock_idx.index(stock_no)]
    index_this_type = []
    edges_weight = []
    for j in range(len(mem)):
        if mem[j] == this_type:
            index_this_type.append(j)
    majority_direction = data_ud[stock_idx.index(stock_no)]

    for j in range(len(mem)):
        if j not in index_this_type:
            edges_weight.append(0.0)
        else:
            diff_day = 0
            for day in range(data_ud.shape[1]):
                if data_ud[j] != majority_direction[day]:
                    diff_day += 1
            edges_weight.append(float(diff_day) / float(data_ud.shape[1]))

    return edges_weight


def read_cluster_info_and_build_graph(
        mem_dir, data_ud_dir, save_data_dir):
    stock_idx_cluster = []
    mem_cluster = []
    with open(mem_dir, 'r') as f:
        lines = csv.reader(f)
        for i, line in enumerate(lines):
            stock_idx_cluster.append(line[0].replace('s', ''))
            mem_cluster.append(int(line[1]))
    with open(data_ud_dir, 'r') as f:
        data_ud_cluster = list(csv.reader(f))
    data_ud_cluster = np.array(data_ud_cluster, dtype=float)
    data_ud_cluster = np.array(data_ud_cluster, dtype=int)

    graph = []
    for stock_no in stock_idx_cluster:
        graph.append(ask_shape_similar(stock_no, mem_cluster, data_ud_cluster, stock_idx_cluster))
    graph = np.array(graph, dtype=float)

    with open(save_data_dir + 'stock_cluster_graph.npy', 'wb') as f:
        np.save(f, graph)

