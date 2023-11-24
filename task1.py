import argparse
from lxml import etree as et


def check_arg_type(arg):
    args = arg.split('=')
    if len(args) != 2:
        raise argparse.ArgumentTypeError('Аргументы введены некорректно.')
    return args[0], args[1]


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('params', nargs='+', type=check_arg_type,
                        help='input=task1.txt output=task1.xml')
    params = dict(parser.parse_args().params)
    return params['input'], params['output']


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
    print(f"Граф создан и сохранен в файл {output_file}.")


def main():
    try:
        input, output = parse_arguments()
    except:
        print("Ошибка чтения аргументов!")
        return 0

    contents = None
    try:
        with open(input) as f:
            contents = f.read()
    except:
        print("Ошибка чтения входного файла.")
        return 0

    try:
        graph = make_graph(contents)
    except:
        print("Не удалось создать граф с помощью входного файла.")
        return 0
    
    try:
        create_xml(graph, output)
    except:
        print("Не удалось сохранить граф в xml-файл.")


if __name__ == "__main__":
    main()
