#!/usr/bin/env python
import click
import logging
import numpy as np
import os
import pandas as pd

from collections import Counter, defaultdict
from copy import deepcopy

# IMPORTANT: this seed cannot be altered!
# It was used for data collection and thus affects the order in which quicksorts are performed in the Just-Sort-It algorithm.
SEED = 2019
LOG = logging.getLogger('results')
TESTSETS = ['pure', 'cross']
METRIC_IDS = ['appropriate', 'informative', 'useful', 'easy_to_answer']


class JustSortIt:
  def __init__(self, n_systems, n_items, rng):
    self.n_systems = n_systems
    self.n_items = n_items
    self.ratings = [list() for _ in range(self.n_items)]
    self.systems = [np.arange(n_systems) for _ in range(self.n_items)]
    self.rng = rng
    for r in self.systems:
      rng.shuffle(r)

  def copeland(self):
    n_wins = np.zeros(self.n_systems)
    for i in range(self.n_items):
      qs = self.qsort(self.systems[i], deepcopy(self.ratings[i]), request_more=False)
      for n in range(0, self.n_systems):
        n_wins[qs[n]] += self.n_systems - n - 1
    return np.argsort(n_wins)

  def dump(self):
    return [self.qsort(self.systems[i], deepcopy(self.ratings[i]), request_more=False) for i in range(self.n_items)]

  def next_rating(self, i):
    try:
      self.qsort(self.systems[i], deepcopy(self.ratings[i]))
    except StopIteration as ex:
      return ex.value
    return None

  def rating_batch(self):
    for i in range(self.n_items):
      r = self.next_rating(i)
      if r:
        yield i, r

  def integrate_ratings(self, ratings):
    for i, (s0, s1, r) in ratings:
      self.ratings[i].append((s0, s1, r))

  def qsort(self, V, ratings, request_more=True, level=0):
    if len(V) < 2:
      return list(V)
    L, R = list(), list()
    p = V[0]  # assume V was shuffled
    for i in V[1:]:
      if not ratings:
        if request_more:
          raise StopIteration((i, p))
        else:
          r = self.rng.binomial(1, 0.5)  # coin flip
      else:
        assert (i, p) == ratings[0][:2]
        r = ratings.pop(0)[2]
      if r:
        L.append(i)
      else:
        R.append(i)
    if level % 2 == 0:
      L = list(self.qsort(L, ratings, request_more=request_more, level=level + 1))
      R = list(self.qsort(R, ratings, request_more=request_more, level=level + 1))
    else:
      R = list(self.qsort(R, ratings, request_more=request_more, level=level + 1))
      L = list(self.qsort(L, ratings, request_more=request_more, level=level + 1))
    return L + [p] + R


def lookup_rating(df, testset, metric, dlg_id, submissionA, submissionB, rng):
  if df is None:
    return None

  subdf = df[((df['Submission1'] == submissionA) & (df['Submission2'] == submissionB)) |
             ((df['Submission1'] == submissionB) & (df['Submission2'] == submissionA))]
  if len(subdf) == 0:
    return None

  subdf = subdf[(subdf['Testset'] == testset) & (
    subdf['Metric'] == metric) & (subdf['DialogueId'] == dlg_id)]

  if len(subdf) == 0:
    return None

  if len(subdf) < 3:
    raise RuntimeError(f"Got <3 rows for {testset}, {metric}, {dlg_id}, {submissionA}, {submissionB}")

  # majority vote, assuming odd # judgements per hit
  AgtB_count = 0

  for _, row in subdf.iterrows():
    A_is_best = (row['Submission1'] == submissionA and row['BestSubmission'] == 1) or (
      row['Submission2'] == submissionA and row['BestSubmission'] == 2)
    AgtB_count += 1 if A_is_best else 0

  if AgtB_count > (len(subdf) / 2):
    return True
  return False


def calculate_rankings(ref_df, n_subs, groups):
  for gname, gindex in groups.items():
    gdf = ref_df.loc[gindex]
    wins = Counter()
    for _, row in gdf.iterrows():
      for r in range(1, n_subs + 1):
        assert not pd.isna(row[f"Rank{r}"])
        wins[row[f"Rank{r}"]] += n_subs - r

    wins_sorted = sorted(wins.items(), key=lambda x: x[1], reverse=True)
    condensed = [[wins_sorted[0][1], wins_sorted[0][0]]]
    for s, c in wins_sorted[1:]:
      if c == condensed[-1][0]:
        condensed[-1].append(s)
      else:
        condensed.append([c, s])
    winner = None
    for item in condensed:
      if len(item) == 2 and item[1] == 'GOLD':
        continue
      winner = item[1:]
      break
    printable = [i[1] if len(i) == 2 else i[1:] for i in condensed]
    LOG.info(f"[{str(gname):35}] WINNER= {str(winner):10}; full ranking= {printable}")


def determine_win_counts(df):
  win_counts = Counter()
  df1 = df[df['BestSubmission'] == 1]
  counts = df1[['BestSubmission', 'Submission1']].groupby('Submission1').count()
  for k, v in counts.iterrows():
    win_counts[k] += v.squeeze()

  df1 = df[df['BestSubmission'] == 2]
  counts = df1[['BestSubmission', 'Submission2']].groupby('Submission2').count()
  for k, v in counts.iterrows():
    win_counts[k] += v.squeeze()

  cmps = Counter()
  for _, row in df.iterrows():
    cmps[row['Submission1']] += 1
    cmps[row['Submission2']] += 1

  normed_win_counts = Counter({k: win_counts[k] / cmps[k] for k in win_counts})

  return win_counts, normed_win_counts


@click.command()
@click.argument('judgements_csv', type=click.Path(exists=True))
@click.option('-d', '--rankings-dump-csv', default=None,
              help='If specified, all rankings per dialogue are dumped to this CSV filepath')
def main(judgements_csv, rankings_dump_csv):
  """./human-evaluation-results.py data/judgements_data.csv -d data/full_rankings.csv"""
  judgements_df = pd.read_csv(judgements_csv)

  subs = sorted(list(set(judgements_df['Submission1']).union(judgements_df['Submission2'])))
  sub_map = {}
  for s in subs:
    if s.endswith('BASELINE'):
      sub_map[s] = 0
      sub_map[0] = s
    elif s == 'GOLD':
      sub_map[s] = 1
      sub_map[1] = s

  non_baseline_subs = deepcopy(subs)
  non_baseline_subs.remove(sub_map[0])
  non_baseline_subs.remove(sub_map[1])
  for i, s in enumerate(non_baseline_subs):
    sub_map[s] = i + 2
    sub_map[i + 2] = s

  for i in range(len(subs)):
    LOG.debug(f"submission map: {sub_map[i].split('/')[-1]} -> {i}")

  next_seed = SEED
  rank_rows = []

  for testset in TESTSETS:
    dlg_id_map = {}
    for i, dlg_id in enumerate(sorted(judgements_df[judgements_df['Testset'] == testset]['DialogueId'].unique())):
      dlg_id_map[i] = dlg_id
      dlg_id_map[dlg_id] = i

    for metric in METRIC_IDS:
      rng = np.random.RandomState(next_seed)
      next_seed += 1
      jsi = JustSortIt(len(subs), len(dlg_id_map) // 2, rng)

      for k in range(100):
        batch = jsi.rating_batch()
        ratings = []
        done = True
        for d, (s1, s2) in batch:
          done = False
          submissionA_zip = sub_map[s1]
          submissionB_zip = sub_map[s2]
          assert s1 != s2
          assert submissionA_zip != submissionB_zip
          dlg_id = dlg_id_map[d]

          rating = lookup_rating(judgements_df, testset, metric, dlg_id, submissionA_zip, submissionB_zip, rng)
          assert rating is not None
          ratings.append((d, (s1, s2, rating)))

        if done:
          break
        jsi.integrate_ratings(ratings)

      LOG.info(f"> DONE COPELAND: {testset.split('-')[-1]} {metric}")
      for i, ranking in enumerate(jsi.dump()):
        row = dict(
            Testset=testset,
            Metric=metric,
            DialogueId=dlg_id_map[i],
          )
        for j, s in enumerate(ranking):
          row[f"Rank{j+1}"] = sub_map[s]
        rank_rows.append(row)

  ranks_df = pd.DataFrame(rank_rows)

  # Rankings
  LOG.info('\n- RANKINGS: OVERALL -------------')
  calculate_rankings(ranks_df, len(subs), dict(ALL=pd.RangeIndex(len(ranks_df))))

  LOG.info('\n- RANKINGS: PER METRIC -------------')
  calculate_rankings(ranks_df, len(subs), ranks_df.groupby('Metric').groups)

  LOG.info('\n- RANKINGS: PER TESTSET -------------')
  calculate_rankings(ranks_df, len(subs), ranks_df.groupby('Testset').groups)

  LOG.info('\n- RANKINGS: PER TESTSET,METRIC -------------')
  calculate_rankings(ranks_df, len(subs), ranks_df.groupby(['Testset', 'Metric']).groups)

  # Win rate
  win_counts, normed_win_counts = determine_win_counts(judgements_df)
  LOG.info('\n- WINS: OVERALL')
  for system, normed_wins in sorted(normed_win_counts.items(), key=lambda x: x[1], reverse=True):
    LOG.info(f"{system:12} - {normed_wins:.2%} won; {win_counts[system]} wins total")

  if rankings_dump_csv:
    LOG.info(f"Dumping ranking results to {rankings_dump_csv}")
    ranks_df.to_csv(rankings_dump_csv, index=False)


if __name__ == '__main__':
  logging.basicConfig(level=logging.DEBUG)
  main()
