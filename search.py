"""
Contains several search algorithms and heuristics to experiment with.

Please note: you will need to provide a usage to
"""

import argparse
import os
from math import sqrt
import heapq
import itertools

parser = argparse.ArgumentParser()

OKBLUE = '\033[94m'
CYAN = '\033[96m'
GREEN = '\033[92m'
RED = '\033[91m'
END = '\033[0m'

parser.add_argument("-method", "--argmethod", required=True)
parser.add_argument("-heuristic", "--argheuristic", required=False)
parser.add_argument('inputfile', metavar='inputfilename', type=str)

args = parser.parse_args()

input_file = args.inputfile
method_str = args.argmethod
heuristic_str = args.argheuristic

if not os.path.exists(input_file):
    print(f"{input_file} was not found. Exiting...")
    exit(1)

sy, sx, gy, gx, grid = None, None, None, None, []

# Transform input file to 2d list
for line in open(input_file):
    row = []
    for c in line:
        if c == 'S':
            sy = len(grid)
            sx = len(row)
            row.append(".")
        elif c == 'G':
            gy = len(grid)
            gx = len(row)
            row.append(".")
        elif c == '%':
            row.append('%')
        elif c == ' ':
            row.append('.')
        elif c.isspace():
            continue
        else:
            print("Input file contains invalid characters. Exiting...")
            exit(1)
    grid.append(row)

# Define methods, global variables

ans_expanded = 0
ans_cost = 0
N, M = len(grid), len(grid[0])
can_traverse = lambda y, x: 0 <= y < N and 0 <= x < M and (grid[y][x] == '.' or grid[y][x] == ',')
dirs = [[0, 1, '>'], [0, -1, '<'], [1, 0, 'v'], [-1, 0, '^']]


# Euclidean distance
def euclidean(uy, ux, vy, vx) -> float:
    return sqrt((uy - vy) ** 2 + (ux - vx) ** 2)


# Manhattan distance
def manhattan(uy, ux, vy, vx) -> int:
    return abs(uy - vy) + abs(ux - vx)


# Standard DFS
def dfs(heuristic, sy, sx, gy, gx):
    if heuristic:
        print("Warn: dfs is uninformed and does not use heuristic.")

    def helper(y, x, cost) -> bool:
        global ans_cost, ans_expanded

        ans_expanded += 1
        grid[y][x] = 'X'
        if y == gy and x == gx:
            grid[y][x] = '*'
            ans_cost = cost
            return True

        for y2, x2, d in dirs:
            new_y, new_x = y2 + y, x2 + x
            if can_traverse(new_y, new_x):
                res = helper(new_y, new_x, cost + 1)
                if res:
                    grid[y][x] = GREEN + d + END
                    return True
        return False

    helper(sy, sx, 0)


# Best-first or greedy search.
# This will probably OOM for large input sizes as it has a time/space complexity of O((N*M))^2 - an extra
# O(N*M) tiles are held to represent the exact path for any node in the PQ
def greedy(heuristic, sy, sx, gy, gx):
    if not heuristic:
        print("Greedy is informed and requires heuristic. Exiting...")
        exit(1)

    global ans_expanded, ans_cost
    pq = []
    heapq.heappush(pq, (heuristic(sy, sx, gy, gx), sy, sx, []))
    expanded = 0

    # use a visited set instead of marking grid, so we know which tiles are on priority queue - prevents case
    # where there is an infinite loop between tiles which have the best heuristics
    visited = set()
    while pq:
        h, y, x, path = heapq.heappop(pq)
        visited.add((y, x))

        grid[y][x] = 'X'
        expanded += 1
        if y == gy and x == gx:
            ans_expanded = expanded
            ans_cost = len(path)
            for i, j, d in path:
                grid[i][j] = d
            break

        for y2, x2, d in dirs:
            y3, x3 = y2 + y, x2 + x
            if (y3, x3) not in visited and can_traverse(y3, x3):
                new_path = path + [(y3, x3, GREEN + d + END)]
                heapq.heappush(pq, (heuristic(y3, x3, gy, gx), y3, x3, new_path))


# Bigraph iterative deepening
def iterative_deepening(heuristic, sy, sx, gy, gx):
    if heuristic:
        print("Warn: iterative deepening is uninformed and does not use heuristic.")

    visited = set()

    def helper(y, x, cost, remaining_depth) -> bool:
        global ans_expanded, ans_cost

        ans_expanded += 1
        visited.add((y, x))

        grid[y][x] = 'X'
        if y == gy and x == gx:
            grid[y][x] = '*'
            ans_cost = cost
            return True

        if remaining_depth == 0:
            return False

        for y2, x2, d in dirs:
            new_y, new_x = y2 + y, x2 + x
            if can_traverse(new_y, new_x):
                res = helper(new_y, new_x, cost + 1, remaining_depth - 1)
                if res:
                    grid[y][x] = GREEN + d + END
                    return True
        return False

    for i in itertools.count(start=0, step=1):
        if helper(sy, sx, 0, i):
            # use our visited set to set visited tiles in grid
            for y, x in visited:
                if grid[y][x] == '.':
                    grid[y][x] = RED + 'X' + END
            return

        # we didn't find our path - reset visited
        for i in range(N):
            for j in range(M):
                if grid[i][j] == 'X':
                    grid[i][j] = '.'


# A* search. Because a* with equal edge costs is just best-first, delegate call
def astar(heuristic, sy, sx, gy, gx):
    return greedy(heuristic, sy, sx, gy, gx)


# link arguments to method and heuristic and execute
METHODS = {"depth-first": dfs, "greedy": greedy, "iterative": iterative_deepening, "astar": astar}
HEURISTICS = {"euclidean": euclidean, "manhattan": manhattan}

method, heuristic = None, None

try:
    method = METHODS[method_str]
    heuristic = HEURISTICS[heuristic_str] if heuristic_str else None
except KeyError as e:
    print(f"{e} not found as argument. Exiting...")
    exit(1)

method(heuristic, sy, sx, gy, gx)


# Make output prettier and display results
grid[sy][sx] = CYAN + 'S' + END
grid[gy][gx] = CYAN + 'G' + END
print('\n========= Expanded Tiles and Path=========')
for i in range(N):
    for j in range(M):
        if grid[i][j] == 'X':
            grid[i][j] = RED + 'X' + END
for row in grid:
    print("".join(row))

vs = {GREEN + c + END for c in {'<', '>', 'v', '^'}}
# print("Sum of expanded:" + str(sum(sum(1 if v == RED + 'X' + END else 0 for v in r) for r in grid)))
# print("Sum of path:" + str(sum(sum(1 if v in vs else 0 for v in r) for r in grid)))

print('\n\n========= Path Without Expanded Tiles =========\n')
for i in range(N):
    for j in range(M):
        if grid[i][j] == RED + 'X' + END:
            grid[i][j] = '.'
for row in grid:
    print("".join(row))

print(f"\nExpanded : {ans_expanded}\nCost: {ans_cost}")
print("\n Please note: there may be a small variance in result of expanded/cost due to nuances in definition-\n"
      " what \"visiting\" a node means, if we count start/end nodes, etc. This program defines expansion\n"
      " of node n as considering visiting the neighbors of n, and cost as moving from one node to another -e.g.\n"
      " S and G being 1 space apart will result in a cost of 1.")
