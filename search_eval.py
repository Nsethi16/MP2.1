import math
import sys
import time

import metapy
import pytoml

class InL2Ranker(metapy.index.RankingFunction):
    """
    Create a new ranking function in Python that can be used in MeTA.
    """
    def __init__(self, some_param=1.0):
        self.c = some_param
        # You *must* call the base class constructor here!
        super(InL2Ranker, self).__init__()

    def score_one(self, sd):
        """
        You need to override this function to return a score for a single term.
        For fields available in the score_data sd object,
        @see https://meta-toolkit.org/doxygen/structmeta_1_1index_1_1score__data.html
        """
        c_t_d = sd.doc_term_count
        avgdl = sd.avg_dl
        doc_length = sd.doc_size

        tfn = c_t_d * math.log2(1 + (avgdl/doc_length))

        c_t_q = sd.query_term_weight
        c_t_c = sd.corpus_term_count
        N = sd.num_docs
        # print("tfn term")
        # print((tfn/(tfn + self.c) ))
        score = c_t_q * (tfn/(tfn + self.c) )* math.log2((N+1))

        return score


def load_ranker(cfg_file):
    """
    Use this function to return the Ranker object to evaluate, e.g. return InL2Ranker(some_param=1.0) 
    The parameter to this function, cfg_file, is the path to a
    configuration file used to load the index. You can ignore this for MP2.
    """
    # return InL2Ranker(some_param=9.9)
    return metapy.index.OkapiBM25(k1=0.2,b=0.1,k3=5)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: {} config.toml".format(sys.argv[0]))
        sys.exit(1)

    cfg = sys.argv[1]
    print('Building or loading index...')
    idx = metapy.index.make_inverted_index(cfg)
    ranker = load_ranker(cfg)
    ev = metapy.index.IREval(cfg)

    with open(cfg, 'r') as fin:
        cfg_d = pytoml.load(fin)

    query_cfg = cfg_d['query-runner']
    if query_cfg is None:
        print("query-runner table needed in {}".format(cfg))
        sys.exit(1)

    start_time = time.time()
    top_k = 10
    query_path = query_cfg.get('query-path', 'queries.txt')
    query_start = query_cfg.get('query-id-start', 0)

    query = metapy.index.Document()
    # output_file_path = 'inl2.avg_p.txt'
    output_file_path = 'bm25.avg_p.txt'
    print('Running queries')
    with open(query_path) as query_file, open(output_file_path, 'w') as output_file:
        for query_num, line in enumerate(query_file):
            # if query_num >= 4:
            #     break  # Exit the loop after 4 iterations
            query.content(line.strip())
            results = ranker.score(idx, query, top_k)
            avg_p = ev.avg_p(results, query_start + query_num, top_k)
            print("Query {} average precision: {}".format(query_num + 1, avg_p))
            output_file.write(str(avg_p) + '\n')
    print("Mean average precision: {}".format(ev.map()))
    print("Elapsed: {} seconds".format(round(time.time() - start_time, 4)))
