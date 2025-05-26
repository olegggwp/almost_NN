import subprocess
import random
import numpy as np
import tempfile
import os

# Параметры генерации
MAX_N = 10  # Можно увеличить для более жёстких тестов
MAX_RC = 5
MAX_LEN = 4
MAX_ALPHA = 10
MAX_VAL = 10
NUM_TESTS = 1000

ops = ['tnh', 'rlu', 'mul', 'sum', 'had']

def gen_matrix(r, c):
    return [[random.randint(-MAX_VAL, MAX_VAL) for _ in range(c)] for _ in range(r)]

def print_matrix(mat):
    return '\n'.join(' '.join(str(x) for x in row) for row in mat)

def gen_test():
    N = random.randint(4, MAX_N)
    M = random.randint(2, N // 2)
    K = random.randint(1, min(2, N - M))
    nodes = []
    shapes = []
    # Сначала M переменных
    for _ in range(M):
        r = random.randint(1, MAX_RC)
        c = random.randint(1, MAX_RC)
        nodes.append(('var', r, c))
        shapes.append((r, c))
    # Затем вычислительные вершины
    for i in range(M, N):
        op = random.choice(ops)
        if op == 'tnh':
            x = random.randint(0, i - 1)
            nodes.append(('tnh', x))
            shapes.append(shapes[x])
        elif op == 'rlu':
            alpha = random.randint(1, MAX_ALPHA)
            x = random.randint(0, i - 1)
            nodes.append(('rlu', alpha, x))
            shapes.append(shapes[x])
        elif op == 'mul':
            # Найти две совместимые матрицы
            found = False
            for _ in range(10):
                a = random.randint(0, i - 1)
                b = random.randint(0, i - 1)
                if shapes[a][1] == shapes[b][0]:
                    found = True
                    break
            if not found:
                # fallback: просто копируем предыдущую
                nodes.append(('tnh', 0))
                shapes.append(shapes[0])
                continue
            nodes.append(('mul', a, b))
            shapes.append((shapes[a][0], shapes[b][1]))
        elif op == 'sum' or op == 'had':
            # выбрать len (1..MAX_LEN) совместимых
            candidates = [j for j in range(i) if shapes[j] == shapes[-1]]
            if not candidates:
                candidates = [i - 1]
            l = random.randint(1, min(MAX_LEN, len(candidates)))
            us = random.sample(candidates, l)
            nodes.append((op, l, us))
            shapes.append(shapes[us[0]])
    # K последних вершин — выходы
    outs = list(range(N - K, N))
    # Формируем входные данные
    lines = [f"{N} {M} {K}"]
    for node in nodes:
        if node[0] == 'var':
            lines.append(f"var {node[1]} {node[2]}")
        elif node[0] == 'tnh':
            lines.append(f"tnh {node[1]+1}")
        elif node[0] == 'rlu':
            lines.append(f"rlu {node[1]} {node[2]+1}")
        elif node[0] == 'mul':
            lines.append(f"mul {node[1]+1} {node[2]+1}")
        elif node[0] == 'sum' or node[0] == 'had':
            us = ' '.join(str(x+1) for x in node[2])
            lines.append(f"{node[0]} {node[1]} {us}")
    # M матриц
    for i in range(M):
        r, c = shapes[i]
        mat = gen_matrix(r, c)
        lines.append(print_matrix(mat))
    # K матриц-градиентов
    for i in outs:
        r, c = shapes[i]
        mat = gen_matrix(r, c)
        lines.append(print_matrix(mat))
    return '\n'.join(lines)

def run_test(test_str):
    with tempfile.NamedTemporaryFile('w+', delete=False) as f:
        f.write(test_str)
        f.flush()
        fname = f.name
    try:
        result = subprocess.run(['python3', 'refactored.py'], stdin=open(fname), stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=5)
        return result.returncode, result.stdout.decode(), result.stderr.decode()
    finally:
        os.unlink(fname)

def main():
    for i in range(NUM_TESTS):
        test = gen_test()
        code, out, err = run_test(test)
        if code != 0 or 'Traceback' in err:
            print(f"FAIL on test #{i+1}:")
            print(test)
            print("STDERR:", err)
            print("STDOUT:", out)
            break
        if (i+1) % 10 == 0:
            print(f"{i+1} tests passed...")
    else:
        print(f"All {NUM_TESTS} tests passed!")

if __name__ == '__main__':
    main()
