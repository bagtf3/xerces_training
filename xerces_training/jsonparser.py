import os
import random
import time
import pickle
import pandas as pd
import numpy as np

from chessbot.review import GameViewer, load_game_index, ANALYZE_PKL
from chessbot.utils import format_time


class JSONParser:
    """
    Lightweight parser that reads a list of game json filenames and
    maintains a sliding training buffer that can be topped up and sampled.
    """

    def __init__(
        self,
        files,
        buffer_size=None,
        draw_rate=0.5,
        min_plies=10,
        recycle=True
    ):
        self.files = list(files)
        self.buffer_size = buffer_size or 20000
        self.draw_rate = draw_rate
        self.min_plies = min_plies
        self.recycle = recycle

        self.idx = 0
        self.buffer = []
        self.start_time = time.time()
        self.refill_times = []
        self.sample_times = []
        self.samples_pushed = []
        self.recycles = 0

    def elapsed(self):
        return time.time() - self.start_time

    def append_to_buffer(self, X, M, P, Z, V):
        y = [0.5 * a + 0.5 * b for a, b in zip(Z, V)]
        self.buffer += list(zip(X, M, P, y))

    def top_up_buffer(self):
        start = time.time()
        while len(self.buffer) < self.buffer_size and self.idx < len(self.files):
            game = self.files[self.idx]
            self.idx += 1

            gv = GameViewer(game, sf_df=None)

            if gv.result == 0:
                if random.random() > self.draw_rate:
                    continue

            if len(gv.moves_uci) < self.min_plies:
                continue

            X, M, P, Z, V, R = gv.generate_training_data(
                sf_skip=False, check_boost=0, capture_boost=0
            )
            if not X:
                continue

            self.append_to_buffer(X, M, P, Z, V)

        # self cycle
        if (self.idx == len(self.files)) and self.recycle:
            # stop the current timer
            stop = time.time()
            self.refill_times.append(stop-start)

            # shuffle and start over
            random.shuffle(self.files)
            self.idx = 0
            self.recycles += 1

            # continue topping
            self.top_up_buffer()

        stop = time.time()
        self.refill_times.append(stop-start)

    def sample_epoch(self, epoch_size):
        """
        Faster sampler: shuffle buffer in-place, take first k, then
        drop them via slicing. Returns stacked arrays or None if empty.
        """
        start = time.time()
        k = epoch_size
        n = len(self.buffer)
        if n == 0:
            return None

        take = k if n >= k else n

        random.shuffle(self.buffer)
        sel = self.buffer[:take]

        Xs, Ms, Ps, Ys = zip(*sel)
        Xb = np.stack(Xs, axis=0)
        Mb = np.stack(Ms, axis=0)
        Pb = np.stack(Ps, axis=0)
        Yb = np.stack(Ys, axis=0)

        # remove consumed items
        self.buffer = self.buffer[take:]

        stop = time.time()
        self.sample_times.append(stop-start)
        self.samples_pushed.append(Xb.shape[0])

        return Xb, Mb, Pb, Yb

    def refill_and_sample(self, epoch_size):
        idx = self.top_up_buffer()
        return self.sample_epoch(epoch_size)

    def reset(self, shuffle=True):
        self.idx = 0
        self.buffer = []
        self.refill_times.clear()
        self.sample_times.clear()
        self.samples_pushed.clear()

        if shuffle:
            random.shuffle(self.files)

    def remaining_files(self):
        return len(self.files) - self.idx

    def buffer_len(self):
        return len(self.buffer)

    def report(self):
        """
        Print a compact report of parser activity and timing stats.
        """
        title = "   JSONParser Stats   ".center(40, "#")
        print(title)

        total_up = self.elapsed()
        total_fill = sum(self.refill_times)
        total_sample = sum(self.sample_times)
        total_samples = sum(self.samples_pushed) if self.samples_pushed else 0

        files_read = self.idx
        buf_len = len(self.buffer)
        remaining = self.remaining_files()

        # formatting helpers
        fmt_time = lambda s: format_time(s)
        fmt_num = lambda n: f"{n:,}"

        if total_samples:
            sample_ms = (total_sample / total_samples) * 1000.0
            fill_plus_sample_ms = ((total_fill + total_sample) /
                                   total_samples) * 1000.0
            sample_per_sample = f"{sample_ms:.3f} ms"
            fill_plus_per_sample = f"{fill_plus_sample_ms:.3f} ms"
        else:
            sample_per_sample = "n/a"
            fill_plus_per_sample = "n/a"

        rows = [
            ("up time", fmt_time(total_up)),
            ("fill time (total)", fmt_time(total_fill)),
            ("sample time (total)", fmt_time(total_sample)),
            ("samples pushed", fmt_num(total_samples)),
            ("sample time / sample", sample_per_sample),
            ("(fill+sample) / sample", fill_plus_per_sample),
            ("files read in", fmt_num(files_read)),
            ("buffer size", fmt_num(buf_len)),
            ("index remaining", fmt_num(remaining)),
            ("recycles", fmt_num(self.recycles))
        ]

        max_label = max(len(r[0]) for r in rows)
        max_val = max(len(r[1]) for r in rows)

        for k, v in rows:
            print(f"[jsonp] {k.ljust(max_label)} : {v.rjust(max_val)}")


def check(a):
    if isinstance(a['json_file'], str):
        return os.path.exists(a['json_file'])
    return False


def selfplay_file_scanner(sp_dir, use_only=None, skip=None, min_plies=10, cpl_cutoff=40):
    run_tags = os.listdir(sp_dir)
    if use_only is not None:
        run_tags = [rt for rt in run_tags if rt in use_only]
    
    if skip is not None:
        run_tags = [rt for rt in run_tags if rt not in skip]
    
    all_games = []
    df_list = []
    for rt in run_tags:
        rd = os.path.join(sp_dir, rt)
        if not os.path.exists(os.path.join(rd, "game_index.json")):
            continue
        ag = load_game_index(rd)
        ag = [a for a in ag if check(a)]
        all_games += ag

        pkl = os.path.join(rd, ANALYZE_PKL)
        with open(pkl, "rb") as f:
            prev_run = pickle.load(f)
        
        df_all = prev_run['df_all']
        df_means = prev_run['df_means']
        
        # tidy up CPL
        df_all['clipped_loss'] = np.clip(df_all['loss'], -1000, 1000)
        clipped_cpl = df_all.groupby("game_id")['clipped_loss'].mean()

        df_means['overall_cpl'] = df_means.game_id.map(clipped_cpl)
        df_means['run_tag'] = rt
        df_list.append(df_means)
        del df_all
        del df_means

    df_trim = pd.concat(df_list).drop_duplicates(['game_id']).sort_values("ts")
    meta = pd.DataFrame(all_games).drop_duplicates("game_id")
    if "overall_cpl" in meta.columns:
        del meta['overall_cpl']
    
    meta = meta.query("plies >= @min_plies")
    meta = meta.merge(df_trim[['game_id', 'run_tag', 'overall_cpl']], on='game_id')
    meta = meta.query("overall_cpl <= @cpl_cutoff")
    meta.sort_values("overall_cpl", ascending=False)
    training_games = meta['json_file'].to_list()

    stats = meta.groupby("run_tag").agg(
        games=("game_id", "nunique"),
        avg_overall_cpl=("overall_cpl", "mean"),
        avg_plies=("plies", "mean"),
    ).reset_index()

    stats["games_fmt"] = stats["games"].map("{:,}".format)
    stats["avg_cpl_fmt"] = stats["avg_overall_cpl"].map(lambda v: f"{v:.2f}")
    stats["avg_plies_fmt"] = stats["avg_plies"].map(lambda v: f"{v:.1f}")

    name_w = max(stats["run_tag"].map(len).max(), len("run_tag"))
    games_w = max(len(s) for s in stats["games_fmt"])
    cpl_w = max(len(s) for s in stats["avg_cpl_fmt"])
    plies_w = max(len(s) for s in stats["avg_plies_fmt"])

    header = (
        f"[selfplay] {'run_tag'.ljust(name_w)} : "
        f"{'games'.rjust(games_w)}  "
        f"{'avg_cpl'.rjust(cpl_w)}  "
        f"{'avg_plies'.rjust(plies_w)}"
    )
    print(header)
    print("[selfplay] " + "-" * (len(header) - 6))

    for _, row in stats.iterrows():
        print(
            f"[selfplay] {row['run_tag'].ljust(name_w)} : "
            f"{row['games_fmt'].rjust(games_w)}  "
            f"{row['avg_cpl_fmt'].rjust(cpl_w)}  "
            f"{row['avg_plies_fmt'].rjust(plies_w)}"
        )

    total_games = stats["games"].sum()
    overall_avg_cpl = meta["overall_cpl"].mean()
    overall_avg_plies = meta["plies"].mean()

    print("[selfplay] " + "-" * (len(header) - 6))
    print(
        f"[selfplay] {'TOTAL'.ljust(name_w)} : "
        f"{total_games:,}".rjust(games_w + 1)
        + "  "
        + f"{overall_avg_cpl:.2f}".rjust(cpl_w + 2)
        + "  "
        + f"{overall_avg_plies:.1f}".rjust(plies_w + 2)
    )

    random.shuffle(training_games)
    return training_games

    
