from src.union_find import UnionFind
import random as r
import sys
from src.tool import parse_args, root
from pathlib import Path

class Grid:

    def __init__(self, size):
        self._init(size)

    def _init(self, size):
        self.size = size
        elements = getattr(self, 'elements', None)
        matrix = getattr(self, 'matrix', None)
        if elements is None:
            del elements
        if matrix is None:
            del matrix
        self.elements = UnionFind(size ** 2)
        self.matrix = [[1 for i in range(size)] for j in range(size)]
        self.matrix[0][:] = [0] * size
        self.matrix[size - 1][:] = [0] * size
        for i in range(size):
            self.connect(0, 0, 0, i)
        for i in range(size):
            self.connect(self.size - 1, 0, self.size - 1, i)
        self.opened = 2 * size
        return self

    def getK(self, i, j):
        return i * self.size + j


    def connect(self, i1, j1, i2, j2):
        self.elements.union(self.getK(i1, j1), self.getK(i2, j2))


    def nearby(self, i, j):
        ijs = [{'i': i+1, 'j': j}, {'i': i-1, 'j': j}, {'i': i, 'j': j-1}, {'i': i, 'j': j+1}, {'i': i, 'j': j}]
        for e in ijs:
            if e['i'] < 0:
                e['i'] = 0
            if e['j'] < 0:
                e['j'] = 0
            if e['i'] >= self.size:
                e['i'] = self.size - 1
            if e['j'] >= self.size:
                e['j'] = self.size - 1
        return ijs


    def find(self, i1, j1, i2, j2):
        return self.elements.find(self.getK(i1, j1), self.getK(i2, j2))


    def shoot(self, i, j):
        if self.matrix[i][j] == 1:
            self.opened += 1
        self.matrix[i][j] = 0
        for e in self.nearby(i, j):
            i1 = e['i']
            j1 = e['j']
            if self.matrix[i1][j1] == 0:
                self.connect(i, j, i1, j1)

    def forward(self):
        while not self.find(0, 0, self.size - 1, self.size - 1):
            head_shot = r.randint(0, self.size ** 2 - 1)
            i, j = (int(head_shot / self.size), head_shot % self.size)
            self.shoot(i, j)
        return self


    def __repr__(self):
        graph = []
        for row in self.matrix:
            elements = []
            for item in row:
                if item == 0:
                    elements.append('\u25a1')
                elif item == 1:
                    elements.append('\u25a0')
            graph.append(elements)

        return '\n'.join([' '.join([str(item) for item in row]) for row in graph])
    


def main():
    args = parse_args()
    size = args.size
    verbose = args.verbose
    save_graph = args.save_graph
    g = Grid(size).forward()
    
    if verbose:
        print(g)
    print()
    print(str(g.opened / (g.size ** 2) * 100) + '%')

    if save_graph:

        from src.plot import save_as_pdf
        threshold = []
        def phi(size_of_board: float):
            return threshold[int(size_of_board) - 1]
        
        threshold = []
        for s in range(2, size + 1):
            g = g._init(s)
            threshold.append(g.opened / (g.size ** 2) * 100)

        save_as_pdf(phi, (1, size), filepath=str(Path(root()) / 'data' / 'output'))



if __name__ == '__main__':
    main()