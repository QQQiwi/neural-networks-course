import argparse
from lxml import etree as et
import math
import os


def sum(x, y):
    return x + y


def multiply(x, y):
    return x * y


def exponent(x):
    return math.exp(x)


STRING_TO_OPERATION = {
    "+": "sum",
    "*": "multiply",
    "exp": "exponent"
}


class Node:
    def __init__(self, node, parents=None, child=None) -> None:
        self.node = node
        self.from_nodes = [] if parents is None else parents
        self.to_nodes = [] if child is None else child


def check_arg_type(arg):
    args = arg.split('=')
    if len(args) != 2:
        raise argparse.ArgumentTypeError('Аргументы введены некорректно.')
    return args[0], args[1]


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('params', nargs='+', type=check_arg_type,
                        help='')
    params = dict(parser.parse_args().params)
    return params['graph'], params['ops'], params['output']


def make_graph(input):
    graph = {}
    pairs = input[1:len(input) - 1].split("), (")
    for elems in pairs:
        edges = elems.split(", ")
        from_vertice = edges[0]
        to_vertice = edges[1]
        vertice_n_order = (to_vertice, edges[2])
        int(edges[2]) # для проверки на корректность задачи порядка
        if from_vertice in graph.keys():
            if vertice_n_order in graph[from_vertice]:
                continue
            graph[from_vertice].append(vertice_n_order)
            graph[from_vertice] = sorted(graph[from_vertice],
                                         key=lambda tup: tup[1])
        else:
            graph[from_vertice] = [vertice_n_order]
        
        if to_vertice not in graph.keys():
            graph[to_vertice] = list()
    return graph


def create_xml(graph, output_file):
    xml_tree = et.Element('graph')
    for v in graph.keys():
        et.SubElement(xml_tree, 'vertex').text = v
    for from_v, listed_edges in graph.items():
        for to_v, order in listed_edges:
            arc = et.SubElement(xml_tree, 'arc')
            et.SubElement(arc, 'from').text = from_v
            et.SubElement(arc, 'to').text = to_v
            et.SubElement(arc, 'order').text = order
    xml_tree = et.ElementTree(xml_tree)
    xml_tree.write(output_file, pretty_print=True, encoding='utf-8')


def get_graph_from_xml(xml_graph):
    xml_tree = et.parse(xml_graph)
    xml_tree_root = xml_tree.getroot()

    graph = {}
    nodes = {}
    for vertex in xml_tree_root.iter("vertex"):
        v = vertex.text
        graph[v] = []
        nodes[v] = Node(v)

    for from_vertex, to_vertex, _ in xml_tree_root.iter("arc"):
        from_vertex = from_vertex.text
        to_vertex = to_vertex.text
        graph[from_vertex].append(to_vertex)
        nodes[from_vertex].to_nodes.append(to_vertex)
        nodes[to_vertex].from_nodes.append(from_vertex)
    
    if cycle_check(graph):
        print("В графе обнаружен цикл.")
        raise RuntimeError

    return graph, nodes
 

def cycle_check(graph):
    path = set()
    def visit(vertex):
        path.add(vertex)
        for neighbour in graph.get(vertex, ()):
            if neighbour in path or visit(neighbour):
                return True
        path.remove(vertex)
        return False
    return any(visit(v) for v in graph)


def iterate_through_nodes(cur_node, nodes):
    from_nodes = [iterate_through_nodes(nodes[p], nodes)
                  for p in cur_node.from_nodes]
    return f'{cur_node.node}({", ".join(from_nodes)})'


def get_linear_interpretation(graph):
    graph, nodes = graph
    root = None
    for vertex in graph.keys():
        if not graph[vertex]:
            root = vertex
    
    result = iterate_through_nodes(nodes[root], nodes)
    return result


def evaluate_graph(graph_string, ops):
    graph_string = graph_string.replace("()", "")
    for cur_op in ops:
        operation = None
        if ops[cur_op] in STRING_TO_OPERATION.keys():
            operation = STRING_TO_OPERATION[ops[cur_op]]
        else:
            operation = str(ops[cur_op])
        graph_string = graph_string.replace(cur_op, operation)
    return eval(graph_string)


def main():
    try:
        graph_input, operations_path, output = parse_arguments()
    except:
        print("Ошибка чтения аргументов!")
        return 0

    contents = None
    try:
        with open(graph_input) as f:
            contents = f.read()
    except:
        print("Ошибка чтения входного файла.")
        return 0

    try:
        graph = make_graph(contents)
    except:
        print("Не удалось создать граф с помощью входного файла.")
        return 0
    
    create_xml(graph, "tmp.txt")
    graph = get_graph_from_xml("tmp.txt")
    os.remove("tmp.txt")
    lin_interpretation = get_linear_interpretation(graph)
    
    try:
        with open(operations_path, 'r') as file:
            operations_dict = file.read()
        operations_dict = eval(operations_dict)
        result = evaluate_graph(lin_interpretation, operations_dict)
        with open(output, 'w') as file:
            file.write(str(result))
        print(f"Значение функции, построенной по графу {graph_input} и файлу" +
              f" {operations_path} сохранено в {output}.")
    except:
        print("Ошибка сопоставления операций с описанием графа!")
        return 0
    


if __name__ == "__main__":
    main()

