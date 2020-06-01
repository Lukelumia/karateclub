import os
import numpy as np
import networkx as nx
from gensim.models.word2vec import Word2Vec
from karateclub.utils.walker import RandomWalker
from karateclub.estimator import Estimator
from src.model.helpers import generator_as_iterator
import jsonlines

class DeepWalk(Estimator):
    r"""An implementation of `"DeepWalk" <https://arxiv.org/abs/1403.6652>`_
    from the KDD '14 paper "DeepWalk: Online Learning of Social Representations".
    The procedure uses random walks to approximate the pointwise mutual information
    matrix obtained by pooling normalized adjacency matrix powers. This matrix
    is decomposed by an approximate factorization technique.

    Args:
        walk_number (int): Number of random walks. Default is 10.
        walk_length (int): Length of random walks. Default is 80.
        dimensions (int): Dimensionality of embedding. Default is 128.
        workers (int): Number of cores. Default is 4.
        window_size (int): Matrix power order. Default is 5.
        epochs (int): Number of epochs. Default is 1.
        learning_rate (float): HogWild! learning rate. Default is 0.05.
        min_count (int): Minimal count of node occurences. Default is 1.
    """
    def __init__(self, walk_number=10, walk_length=80, dimensions=128, workers=4,
                 window_size=5, epochs=1, learning_rate=0.05, min_count=1, cached_walks_path=None, cached_prefix='prewalking_deepwalk'):

        self.walk_number = walk_number
        self.walk_length = walk_length
        self.dimensions = dimensions
        self.workers = workers
        self.window_size = window_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.min_count = min_count
        self.cached_walks_path = cached_walks_path
        self.cached_prefix = cached_prefix

    def check_if_walk_exists(self):
        if self.cached_walks_path is None:
            return False

        if not os.path.isdir(self.cached_walks_path):
            return False

        expected_folder_name = f'{self.cached_prefix}_{self.walk_length}_{self.walk_number}'.lower()
        folders = os.listdir(self.cached_walks_path)
        for folder in folders:
            if not folder.lower().startswith(self.cached_prefix):
                continue
            folder_walk_length, folder_walk_number = folder.split(f'{self.cached_prefix}')[1].split('_')
            if int(folder_walk_length) >= self.walk_length and int(folder_walk_number) >= self.walk_number:
                prewalk_file = os.path.join(self.cached_walks_path, folder, 'walks_dump.jsonl')
                if os.path.isfile(prewalk_file):
                    self.prewalk_file = prewalk_file
                    return True
        return False

    def load_prewalk(self):
        # def open_file():
        #     return jsonlines.open(self.prewalk_file).iter()
        def open_file():
            folder_name = os.path.basename(os.path.dirname(self.prewalk_file))
            file_walk_length, file_walk_number = folder_name.split('prewalking_deepwalk_')[1].split('_')
            with jsonlines.open(self.prewalk_file) as reader:
                for num, line in enumerate(reader):
                    if num % int(file_walk_number) < self.walk_number:
                        yield line[:int(self.walk_length)]

        return generator_as_iterator(open_file)

    def fit(self, graph):
        """
        Fitting a DeepWalk model.

        Arg types:
            * **graph** *(NetworkX graph)* - The graph to be embedded.
        """

        if not self.check_if_walk_exists():
            self._check_graph(graph)
            # This creates the sentences and keeps it in memory
            walker = RandomWalker(self.walk_length, self.walk_number)
            walker.do_walks(graph)
            sentences = walker.walks
        else:
            # This returns a iterator
            sentences = self.load_prewalk()

        model = Word2Vec(sentences,
                         hs=1,
                         alpha=self.learning_rate,
                         iter=self.epochs,
                         size=self.dimensions,
                         window=self.window_size,
                         min_count=self.min_count,
                         workers=self.workers)

        # num_of_nodes = graph.number_of_nodes()
        # self._embedding = [model[str(n)] for n in range(num_of_nodes)]

        self._embedding = list()
        x = 0
        while True:
            try:
                self._embedding.append(model[str(x)])
                x+=1
            except:
                print('found {x} consecutive nodes')
                break


    def get_embedding(self):
        r"""Getting the node embedding.

        Return types:
            * **embedding** *(Numpy array)* - The embedding of nodes.
        """
        return np.array(self._embedding)
