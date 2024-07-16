"""A utility that aggregates partial bfs results.

Requirements:
  Python 3.6.8 or above

Usage:

  $ cat bfs_results_0-31.txt bfs_results_32-63.txt | python ./aggregate_bfs_results.py
  or
  $ python ./aggregate_bfs_results.py bfs_results_0-31.txt bfs_results_32-63.txt
"""

import itertools
import math
import re
import statistics
import sys
from typing import Dict, List, NamedTuple

BFS_TIME_REGEX = re.compile(r"Time for BFS (\d+) is ([\d\.]+)(?: \(([\d\.e\+\-]+)\))?")
BFS_VALIDATION_TIME_REGEX = re.compile(r"Validate time for BFS (\d+) is ([\d\.]+)")
BFS_TRAVERSED_EDGES_REGEX = re.compile(r"Number of traversed edges is (\d+)$")


def main(files: List[str]) -> None:
    if not files or files == ["-"]:
        reader = sys.stdin
    else:
        reader = itertools.chain(*[open(f, "r") for f in files])

    results: Dict[int, "BFSResult"] = {}
    for line in reader:
        m = BFS_TIME_REGEX.search(line)
        if not m:
            continue
        bfs_idx = int(m.group(1))
        if bfs_idx in results:
            raise Exception("duplicated bfs index")
        bfs_time = float(m.group(3) if m.group(3) is not None else m.group(2))

        m = BFS_VALIDATION_TIME_REGEX.search(next(reader))
        if not m:
            raise Exception("invalid validation time line")
        if bfs_idx != int(m.group(1)):
            raise Exception("invalid validation bfs index")
        validation_time = float(m.group(2))

        m = BFS_TRAVERSED_EDGES_REGEX.search(next(reader))
        if not m:
            raise Exception("invalid traversed edges line")
        n_traversed_edges = int(m.group(1))

        results[bfs_idx] = BFSResult(
            bfs_time=bfs_time,
            validation_time=validation_time,
            n_traversed_edges=n_traversed_edges,
            bfs_teps=n_traversed_edges / bfs_time,
        )

    if min(results.keys()) != 0 or max(results.keys()) + 1 != len(results):
        print("[WARNING] This is partial results")

    n_bfs_root = len(results)
    print_result("time", [r.bfs_time for r in results.values()], n_bfs_root)
    print_result("nedge", [r.n_traversed_edges for r in results.values()], n_bfs_root)
    print_result(
        "TEPS",
        [r.bfs_time / r.n_traversed_edges for r in results.values()],
        n_bfs_root,
        True,
    )
    print_result("validate", [r.validation_time for r in results.values()], n_bfs_root)


def print_result(
    suffix: str, data: List[float], n_bfs_root: int, harmonic_mean_mode=False
) -> None:
    data.sort()
    n = len(data)
    spaces = " " * (12 - len(suffix))

    v_min, v_max = data[0], data[-1]
    v_quartile_1 = (data[(n - 1) // 4] + data[n // 4]) * 0.5
    v_median = (data[(n - 1) // 2] + data[n // 2]) * 0.5
    v_quartile_3 = (data[n - 1 - (n - 1) // 4] + data[n - 1 - (n // 4)]) * 0.5

    if harmonic_mean_mode:
        v_min, v_max = 1 / v_max, 1 / v_min
        v_quartile_1, v_quartile_3 = 1 / v_quartile_3, 1 / v_quartile_1
        v_median = 1 / v_median

    print("min_{}:               {}{:.12g}".format(suffix, spaces, v_min))
    print("firstquartile_{}:     {}{:.12g}".format(suffix, spaces, v_quartile_1))
    print("median_{}:            {}{:.12g}".format(suffix, spaces, v_median))
    print("thirdquartile_{}:     {}{:.12g}".format(suffix, spaces, v_quartile_3))
    print("max_{}:               {}{:.12g}".format(suffix, spaces, v_max))
    if harmonic_mean_mode:
        v_harmonic_mean = 1 / statistics.mean(data)
        tmp_mean = statistics.mean(data)
        tmp_stdev = statistics.stdev(data)
        v_harmonic_stdev = tmp_stdev / (tmp_mean * tmp_mean * math.sqrt(n_bfs_root - 1))
        print("harmonic_mean_{}:     {}{:.12g}".format(suffix, spaces, v_harmonic_mean))
        print(
            "harmonic_stddev_{}:   {}{:.12g}".format(suffix, spaces, v_harmonic_stdev)
        )
    else:
        v_mean = statistics.mean(data)
        v_stdev = statistics.stdev(data)
        print("mean_{}:              {}{:.12g}".format(suffix, spaces, v_mean))
        print("stddev_{}:            {}{:.12g}".format(suffix, spaces, v_stdev))


class BFSResult(NamedTuple):
    bfs_time: float
    validation_time: float
    n_traversed_edges: int
    bfs_teps: float


if __name__ == "__main__":
    main(sys.argv[1:])
