from lxml import etree as et
import argparse


def check_arg_type(arg):
    args = arg.split('=')
    if len(args) != 2:
        raise argparse.ArgumentTypeError('Аргументы введены некорректно.')
    return args[0], args[1]


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('params', nargs='+', type=check_arg_type,
                        help='input=task1.xml output=task2.txt')
    params = dict(parser.parse_args().params)
    return params['input'], params['output']


class Node:
    def __init__(self, node, parents=None, child=None) -> None:
        self.node = node
        self.from_nodes = [] if parents is None else parents
        self.to_nodes = [] if child is None else child


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


def get_linear_interpretation(graph, output):
    graph, nodes = graph
    root = None
    for vertex in graph.keys():
        if not graph[vertex]:
            root = vertex
    
    result = iterate_through_nodes(nodes[root], nodes)
    with open(output, 'w') as file:
        file.write(result)
    print(f"Линейное представление функции сохранено в файл {output}.")


def main():
    try:
        input, output = parse_arguments()
    except:
        print("Ошибка чтения аргументов!")
        return 0
    
    try:
        graph = get_graph_from_xml(input)
    except:
        print("Не удалось считать граф из входного файла.")
        return 0
    
    try:
        get_linear_interpretation(graph, output)
    except:
        print("Не удалось получить линейное представление функции.")



if __name__ == "__main__":
    main()