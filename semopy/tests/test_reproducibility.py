# https://gitlab.com/georgy.m/semopy/-/issues/26#note_913412910

from semopy.examples import political_democracy as pd
from multiprocessing import Process, Queue

import numpy as np
import random
import semopy


def fit_model() -> np.array:
    random.seed(0)
    np.random.seed(0)

    desc = pd.get_model()
    data = pd.get_data()

    m = semopy.Model(desc)
    return m.fit(data).x


n_tries = 2


def test_reproducible_same_process():
    # Within the same process/BLAS thread pool, results are bit-for-bit
    # reproducible given the same seed.
    results = [fit_model() for _ in range(n_tries)]
    assert np.isclose(*results, atol=1e-16, rtol=0).all()


def fit_model_async(q: Queue) -> None:
    q.put(fit_model())


def test_reproducible_different_processes():
    # Across process boundaries, the underlying BLAS library may pick a
    # different number of threads (and therefore a different floating-point
    # reduction order), so results can differ by a small amount even with
    # identical seeds. A looser tolerance than the same-process case is used
    # to check for reproducibility up to numerical noise rather than
    # bit-exact equality.
    q = Queue()
    processes = [Process(target=fit_model_async, args=(q,)) for _ in range(n_tries)]
    for p in processes:
        p.start()
        p.join()
    results = [q.get() for _ in range(n_tries)]
    assert np.isclose(*results, atol=1e-8, rtol=1e-6).all()
