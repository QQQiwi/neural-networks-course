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
                        help='input=arcs1.txt output=output1.xml')
    params = dict(parser.parse_args().params)
    return params['input'], params['output']


def parse_input(input):
    graph = None

    pairs = input[1:len(input) - 1].split("), (")
    for elems in pairs:
        pass

    return graph


def create_xml(graph):
    pass


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
        print("Ошибка чтения входного файла!")
        return 0

    try:
        graph = parse_input(contents)
    except:
        print("Ошибка формата записи данных во входном файле!")
        return 0
    # create_xml(graph)


if __name__ == "__main__":
    main()
