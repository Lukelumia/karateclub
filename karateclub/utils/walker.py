import os
import random
import networkx as nx
from tqdm.auto import tqdm
import jsonlines

class RandomWalker:
    """
    Class to do fast first-order random walks.

    Args:
        walk_length (int): Number of random walks.
        walk_number (int): Number of nodes in truncated walk.
    """
    def __init__(self, walk_length, walk_number, silent=False, dump_path=None, dump_size=100000):
        self.walk_length = walk_length
        self.walk_number = walk_number
        self.silent = silent
        self.dump_path = dump_path
        self.dump_size = dump_size

    def do_walk(self, node):
        """
        Doing a single truncated random walk from a source node.

        Arg types:
            * **node** *(int)* - The source node of the diffusion.

        Return types:
            * **walk** *(list of strings)* - A single truncated random walk.
        """
        walk = [node]
        for _ in range(self.walk_length-1):
            nebs = self.neighbors[walk[-1]]
            if len(nebs) > 0:
                walk = walk + random.sample(nebs, 1)
        walk = [str(w) for w in walk]
        return walk

    def dump_walks(self):
        dump_path = os.path.join(self.dump_path, f'walks_dump.jsonl')
        with jsonlines.open(dump_path, 'a') as writer:
            writer.write_all(self.walks)
        self.walks = list()

    def do_walks(self, graph):
        """
        Doing a fixed number of truncated random walk from every node in the graph.

        Arg types:
            * **graph** *(NetworkX graph)* - The graph to run the random walks on.
        """
        self.walks = []
        self.graph = graph
        self.neighbors = dict()
        for node in graph.nodes():
            self.neighbors[node] = list(graph.neighbors(node))

        if self.silent:
            iter = self.graph.nodes()
        else:
            iter = tqdm(self.graph.nodes(), desc=f'Generating walks ({self.walk_number} per node)')
        for num, node in enumerate(iter):
            for _ in range(self.walk_number):
                walk_from_node = self.do_walk(node)
                self.walks.append(walk_from_node)

            if (num + 1) % self.dump_size == 0:
                if self.dump_path is not None:
                # If we want to dump the results do it every {dump_size}
                    self.dump_walks()
        if self.dump_path is not None:
            # Also make sure the end is saved in the last batch
            self.dump_walks()
