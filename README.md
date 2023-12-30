# Практика по предмету "Нейронные сети"

## Задание 1

Пример тестового запуска:

```
python task1.py input=tests\test1.txt output=tests\task1.xml
```

## Задание 2

Пример тестового запуска:

```
python task2.py input=tests\task1.xml output=tests\task2.txt
```

## Задание 4

Пример тестового запуска:

```
python .\task4.py x=x.txt w=w.txt nn=kekw.txt y=out.txt
```

Для запуска без ввода начальных значений весов (тогда веса сгенерируются автоматически исходя из количества значений $x$):

```
python .\task4.py x=x.txt w=None nn=kekw.txt y=out.txt
```

## Задание 5

Пример тестового запуска (без ввода начальных значений весов):

```
python task5.py x=x5.txt y=y.txt w=None nn=why.txt epochs=1000 loss=results.txt
```

где аргумент **loss** - путь, в котором будет создан файл со значениями функции потерь на каждой 10-й эпохе, аргумент **nn** - путь, в котором будет создан файл со значениями весов после процесса обучения, аргумент **epochs** задает количество эпох обучения.

Скорость спуска (learning rate) в данной реализации задан со значением по умолчанию $0.1$.