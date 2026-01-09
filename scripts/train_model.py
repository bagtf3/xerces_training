import os, sys, yaml

import numpy as np
from xerces_training.chunkparser import ChunkParser
from xerces_training.jsonparser import JSONParser, selfplay_file_scanner

from chessbot import SP_DIR
from chessbot.utils import batch_policy_metrics, print_validation, format_time
from chessbot.model import load_model

import pandas as pd

import matplotlib.pyplot as plt
import time


def moving_average_pd(arr, window=15):
    s = pd.Series(arr)
    return s.rolling(window, center=True, min_periods=1).mean().values


def load_training_config(path):
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    return cfg


def get_loss_weights(weight_dict, epoch, current_loss_weights):
    for k in sorted(weight_dict.keys()):
        if epoch < k:
            weight = weight_dict[k]
            break

    for k in current_loss_weights.keys():
        current_loss_weights[k] = weight
    return current_loss_weights


def combine_batches(lc0_batch, selfplay_batch, shuffle=True):
    """
    Combine lc0 and selfplay batches into a single training batch.
    Returns Xb, Mb, Pb, Yb, fens (fens is a list, may contain empty strings).
    """
    # both empty
    if lc0_batch is None and selfplay_batch is None:
        return None

    # only selfplay
    if lc0_batch is None:
        Xb, Mb, Pb, Yb = selfplay_batch
        if shuffle:
            perm = np.random.permutation(Xb.shape[0])
            return Xb[perm], Mb[perm], Pb[perm], Yb[perm]
        return Xb, Mb, Pb, Yb

    # only lc0
    if selfplay_batch is None:
        tokens, masks, policies, ys = lc0_batch
        if shuffle:
            perm = np.random.permutation(tokens.shape[0])
            return tokens[perm], masks[perm], policies[perm], ys[perm]
        return tokens, masks, policies, ys

    # both present: unpack
    tokens, masks, policies, ys = lc0_batch
    Xb2, Mb2, Pb2, Yb2 = selfplay_batch

    # concat along batch axis
    Xb = np.concatenate([tokens, Xb2], axis=0)
    Mb = np.concatenate([masks, Mb2], axis=0)
    Pb = np.concatenate([policies, Pb2], axis=0)
    Yb = np.concatenate([ys, Yb2], axis=0)

    if shuffle:
        perm = np.random.permutation(Xb.shape[0])
        Xb = Xb[perm]
        Mb = Mb[perm]
        Pb = Pb[perm]
        Yb = Yb[perm]

    return Xb, Mb, Pb, Yb


def run_training(cfg):
    # start unpacking
    run_dir = cfg['run_dir']
    model_path = cfg['model_path']
    batch_size = cfg['batch_size']
    major_batch_size = batch_size * cfg['major_batch_mult']
    lc0_batch_size = 0
    selfplay_batch_size = 0
    
    start = time.time()
    def elapsed():
        return format_time(time.time() - start)
    
    model = load_model(model_path)
    backup_path = str(model_path).split(".h5")[0] + ".bak.h5"
    model.save(backup_path)
    print(f"[training] Model loaded and backed up {elapsed()}")
    
    # assumes always using Lc0 data
    lc0_dir = cfg['lc0_dir']
    sub_dirs = [d for d in os.listdir(lc0_dir)]
    sub_dirs = [d for d in sub_dirs if os.path.isdir(os.path.join(lc0_dir, d))]
    lc0_runs = cfg['lc0_runs']

    if isinstance(lc0_runs, list):
        sub_dirs = [d for d in sub_dirs if d in lc0_runs]
    elif str(lc0_runs).lower() == 'all':
        pass
    else:
        raise Exception(f"Unknown lc0_runs parameter {lc0_runs}")
    
    chunks = []
    counts = {}
    for d in sub_dirs:
        full_dir = os.path.join(lc0_dir, d)
        if not os.path.isdir(full_dir):
            continue
        gz_files = [f for f in os.listdir(full_dir) if f.endswith(".gz")]
        counts[d] = len(gz_files)
        full_paths = [os.path.join(full_dir, f) for f in gz_files]
        chunks.extend(full_paths)

    # drop duplicates while preserving order
    chunks = list(dict.fromkeys(chunks))

    max_name = max((len(d) for d in sub_dirs), default=5)
    max_count = max((counts.get(d, 0) for d in sub_dirs), default=0)
    count_w = len(f"{max_count:,}") + 1
    
    print("[training] Building Lc0 parser")
    for d in sub_dirs:
        c = counts.get(d, 0)
        print(f"[lc0] {d.ljust(max_name)} : {f'{c:,}'.rjust(count_w)} files")

    print(f"[lc0] {'TOTAL'.ljust(max_name)} : "
        f"{f'{len(chunks):,}'.rjust(count_w)} files")

    shuffle_size = min(batch_size*cfg['shuffle_size_bs_mult'], 20000)
    lc0_mix = cfg.get("lc0_to_selfplay_mix", 1.0)
    if lc0_mix == 1.0:
        lc0_batch_size = int(major_batch_size)
    else:
        lc0_batch_size = int(np.floor(major_batch_size*lc0_mix))
        selfplay_batch_size = int(major_batch_size - lc0_batch_size)
    
    lc0_parser = ChunkParser(
        chunks,
        shuffle_size=shuffle_size,
        sample=cfg["sample"],
        batch_size=lc0_batch_size,
        workers=cfg["workers"],
        diff_focus_min=cfg.get("diff_focus_min", 0.5),
        diff_focus_slope=cfg.get("diff_focus_slope", 0.15),
        diff_focus_q_weight=cfg.get("diff_focus_q_weight", 3.0),
        diff_focus_pol_scale=cfg.get("diff_focus_pol_scale", 2.0),
    )
    
    json_parser = None
    if cfg['use_selfplay_data']:
        # check to see if anything was set for Lc0
        if lc0_batch_size == 0:
            selfplay_batch_size = major_batch_size

        use_only = None    
        if cfg.get("selfplay_runs", 'all') != 'all':
            use_only = cfg['selfplay_runs']

        skip = cfg.get("selfplay_skip", [])
        min_plies = cfg.get("min_plies", 10)
        print("[training] Building JSON selfplay parser")
        sp_game_files = selfplay_file_scanner(
            sp_dir=cfg['selfplay_dir'], use_only=use_only, skip=skip,
            min_plies=min_plies,
            cpl_cutoff=cfg.get("selfplay_cp_threshold", 35)
        )

        js_buffer_size = batch_size*cfg.get("jsonparser_buffer_mult", 30)
        draw_rate = cfg.get("draw_rate", 0.5)
        json_parser = JSONParser(sp_game_files, js_buffer_size, draw_rate, min_plies)
        
    print(f"[training] Parsers Built {elapsed()}")
    
    # set number of epoch and loss weights
    n_epochs = cfg['epochs']
    lw = cfg['loss_weights']

    # weight duration
    wd = n_epochs // (len(lw) + 1)
    weight_dict = {wd*(i+1):w for i, w in enumerate(lw)}
    
    # training run starts here
    metrics_history = {"value_mse": [], "value_corr": []}
    epoch_time_list = []

    eps = 1e-12
    big_neg = -1e6
    
    progress_file = cfg['progress_file']
    if not os.path.exists(progress_file):
        eval_df = None
    else:
        eval_df = pd.read_csv(progress_file)

    loss_weights = {'policy_logits': 1.0, 'value_out': 1.0}
    print(f"[training] Starting training run {elapsed()}")
    validate_every = cfg.get('validate_every', 10)
    begin = time.time()
    
    for step, lc0_batch in enumerate(lc0_parser.sequential()):
        epoch_start = time.time()
        if step % 17 == 0:
            lc0_parser.report()
            
        epoch = step*10
        if epoch > n_epochs:
            break

        loss_weights = get_loss_weights(weight_dict, epoch, loss_weights)
        
        # optionally mix in seflplay data
        if json_parser is not None:
            if step % 15 == 0:
                json_parser.report()
            for attempt in range(2):
                try:
                    selfplay_batch = json_parser.refill_and_sample(selfplay_batch_size)
                    break
                except Exception as e:
                    print(f"[train] json_parser error (attempt {attempt+1}): {e}")
                    json_parser.reset()
                    # re-raise on final attempt to surface the real error
                    if attempt == 1:
                        raise
        else:
            selfplay_batch = None
        
        # combine these
        train_batch = combine_batches(lc0_batch, selfplay_batch, shuffle=True)
        if train_batch is None:
            continue

        Xb, Mb, Pb, Yb = train_batch
        if step % validate_every == 0:
            # pred validate and train
            preds = model.predict(Xb, verbose=0, batch_size=batch_size)
            value_preds = preds[1].ravel(); targets = np.asarray(Yb).ravel()
            
            # save true vs pred scatter to disk (don't show)
            tvp_path = os.path.join(run_dir, "true_vs_pred_plot_latest.png")
            fig = plt.figure(figsize=(6, 6))
            ax = fig.add_subplot(1, 1, 1)
            ax.scatter(Yb, value_preds, s=6)
            ax.plot([-1, 1], [-1, 1], linestyle="--", color="red", alpha=0.6)
            ax.set_xlim(-1, 1); ax.set_ylim(-1, 1)
            ax.set_xlabel("target"); ax.set_ylabel("pred")
            ax.set_title("pred vs target")
            fig.tight_layout()
            fig.savefig(tvp_path)
            plt.close(fig)

            policy_logits = preds[0]  # (B,4096)
            value_preds = preds[1].ravel()  # (B,)
        
            policy_stats = batch_policy_metrics(policy_logits, Pb, Mb)
        
            targets = np.asarray(Yb).ravel()
            value_mse = np.mean((value_preds - targets) ** 2)
            value_corr = np.corrcoef(value_preds, targets)[0, 1]
        
            for k, v in policy_stats.items():
                if k not in metrics_history:
                    metrics_history[k] = []
                
                metrics_history[k].append(v)
            
            metrics_history['value_mse'].append(value_mse)
            metrics_history['value_corr'].append(value_corr)
        
            # create eval_df row
            latest_row = pd.DataFrame(metrics_history).iloc[[-1], :]
            n_samples = Xb.shape[0]
            if step > 0:
                n_samples *= validate_every
            latest_row['n_samples'] = n_samples        
            
            # check for updates
            if os.path.exists(cfg['progress_file']):
                eval_df = pd.read_csv(progress_file)
                
            if eval_df is None:
                latest_row['model_epoch'] = 0
                eval_df = latest_row.copy()
            else:
                latest_row['model_epoch'] = eval_df.tail(1)["model_epoch"].item() + 1
                latest_row = latest_row.reindex(columns=eval_df.columns)
                eval_df = pd.concat([eval_df, latest_row])
            
            eval_df.to_csv(progress_file, index=False)
        
            # build print dict and call the printer
            print_metrics = {"value_mse": value_mse, "value_corr": value_corr}
            print_metrics.update(policy_stats)
            print_validation(step, print_metrics)
            
            retrains = eval_df.tail(1)['model_epoch'].item()
            hide_first = max(int(0.1 * retrains), 5)
            ma_window = min(max(3, int(retrains * 0.2)), 15)
            if ma_window % 2 == 0:
                ma_window += 1
        
            x = np.arange(len(eval_df["policy_ce"]))
            start = hide_first
            xs = x[start:]
            if len(xs) > hide_first:
                fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            
                # policy CE
                ax = axes[0, 0]
                raw = np.array(eval_df["policy_ce"])[start:]
                ma = moving_average_pd(eval_df["policy_ce"], window=ma_window)[start:]
                if raw.size:
                    ax.plot(xs, raw, label="policy_ce", alpha=0.6, lw=1)
                if ma.size:
                    ax.plot(xs, ma, label=f"MA{ma_window}", lw=2)
                ax.set_title("policy CE (nats)")
                ax.legend()
            
                # CE gain vs uniform
                ax = axes[0, 1]
                raw = np.array(eval_df["ce_gain"])[start:]
                ma = moving_average_pd(eval_df["ce_gain"], window=ma_window)[start:]
                if raw.size:
                    ax.plot(xs, raw, label="ce_gain", alpha=0.6, lw=1)
                if ma.size:
                    ax.plot(xs, ma, label=f"MA{ma_window}", lw=2)
                ax.set_title("CE gain vs uniform")
                ax.legend()
            
                # value MSE
                ax = axes[1, 0]
                raw = np.array(eval_df["value_mse"])[start:]
                ma = moving_average_pd(eval_df["value_mse"], window=ma_window)[start:]
                if raw.size:
                    ax.plot(xs, raw, label="mse", alpha=0.6, lw=1)
                if ma.size:
                    ax.plot(xs, ma, label=f"MA{ma_window}", lw=2)
                ax.set_title("value MSE")
                ax.legend()
            
                # value corr
                ax = axes[1, 1]
                raw = np.array(eval_df["value_corr"])[start:]
                ma = moving_average_pd(eval_df["value_corr"], window=ma_window)[start:]
                if raw.size:
                    ax.plot(xs, raw, label="corr", alpha=0.6, lw=1)
                if ma.size:
                    ax.plot(xs, ma, label=f"MA{ma_window}", lw=2)
                ax.set_title("value corr")
                ax.legend()
            
                plt.tight_layout()
                # save eval progress plot
                ep_path = os.path.join(run_dir, "eval_progess.png")
                fig.tight_layout()
                fig.savefig(ep_path)
                plt.close(fig)

        # main fit
        Ydict = {"value_out": Yb.astype(np.float32), "policy_logits": Pb}
        weights = np.ones_like(Yb)
        s_wts = {k: weights*loss_weights[k] for k in Ydict.keys()}
        print("-"*100)
        history = model.fit(
            Xb, Ydict, epochs=1, batch_size=batch_size, verbose=0, sample_weight=s_wts
        )
        
        rows = []
        for m, v in history.history.items():
            name = "total" if m == "loss" else m.replace("_loss", "")
            start = v[0]; end = v[-1]
            delta = start - end
            mark = "*" if delta < 0 else "+"
            rows.append((name, start, end, delta, mark))
        
        name_w = max(len(r[0]) for r in rows)
        num_w = 8   # width for numbers (including decimal point)
        ETAG = f"[epoch {step:4d}]"
        fmt = (f"{ETAG} [model fit] "
            f"{{name:<{name_w}}} : value: {{start:{num_w}.4f}} -> "
            f"{{end:{num_w}.4f}}  delta: {{delta:{num_w}.4f}} {{mark}}")

        for name, start, end, delta, mark in rows:
            print(fmt.format(name=name, start=start, end=end, delta=delta, mark=mark))
                    
        epoch_time = time.time() - epoch_start
        epoch_time_list.append(epoch_time)
        e_time = format_time(epoch_time)
        runtime = format_time(time.time() - begin)
        avg_epoch = format_time(np.mean(epoch_time_list))

        if step % cfg.get("checkpoint_every", 50) == 0:
            print("[training] Checkpointing model")
            model.save(cfg['model_file'])
        
        print("-"*100)
        print(f"[time check] last epoch: {e_time}, avg epoch: {avg_epoch},  "
            f"total runtime: {runtime}")
        print()

    # when done
    print(f"[training] Training Complete {elapsed}")
    model.save(cfg['model_file'])

    
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python train_model.py <run_tag>")
        sys.exit(1)

    run_tag = sys.argv[1]
    run_dir = os.path.join(SP_DIR, run_tag)

    if not os.path.isdir(run_dir):
        print(f"[error] run dir not found: {run_dir}")
        sys.exit(1)

    # find run config yaml
    yaml_path = None
    for nm in ("training_config.yaml", "training_config.yml"):
        p = os.path.join(run_dir, nm)
        if os.path.exists(p):
            yaml_path = p
            break
        
    if yaml_path is None:
        print(f"[error] no training_config.yaml found in {run_dir}")
        sys.exit(1)

    # load base config and validation configs
    cfg = load_training_config(yaml_path)
    cfg['run_dir'] = run_dir
    cfg['run_tag'] = run_tag
    
    # check for the model
    model_file = run_tag + "_model.h5"
    model_path = os.path.join(run_dir, model_file)
    if not os.path.exists(model_path):
        print(f"[error] no {model_file} found in {run_dir}")
        sys.exit(1)
        
    cfg['model_path'] = model_path
    cfg['progress_file'] = os.path.join(run_dir, "eval_progress.csv")
    model = run_training(cfg)

