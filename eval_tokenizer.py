#!/usr/bin/env python3
"""
Tokenizer reconstruction evaluation for SMART.

Measures how well the 2048-token vocabulary reconstructs ground-truth agent
trajectories by comparing decoded token bounding boxes to ground-truth boxes.

Both sides use the same canonical bbox dimensions (the tokenizer never reads
actual agent dimensions), so the metric isolates motion-quantization error.

Three reconstruction modes are reported:

  endpoint   — compare token_contour (matched endpoint polygon) against the
               GT bbox at each 5-frame token boundary.  Uses the output of
               TokenProcessor directly; no additional decoding needed.

  dense_indep — decode the full 6-frame token_all sequence for each window,
               anchored to the ACTUAL trajectory pose at the window start.
               Compares at all 10 Hz frames.  Measures per-token error in
               isolation (no error accumulation across windows).

  dense_chain — same decoding, but each window is anchored to the EXIT POSE
               of the previous token (mirroring match_token and inference).
               Accumulates chaining errors across the trajectory.

Usage:
    python eval_tokenizer.py --data_dir ./data/waymo_processed/validation
    python eval_tokenizer.py --data_dir ./data/waymo_processed/validation \\
        --num_scenarios 200 --output_csv results/tokenizer_eval.csv
"""

import sys
import pickle
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))

from smart.datasets.preprocess import TokenProcessor, cal_polygon_contour


# ── Constants ─────────────────────────────────────────────────────────────────

TYPE_TO_CAT   = {0: 'veh', 1: 'ped', 2: 'cyc'}
CAT_DIMS      = {'veh': (2.0, 4.8), 'cyc': (1.0, 2.0), 'ped': (1.0, 1.0)}
MOVING_THRESH = 0.5   # m/s — below this the agent is labelled 'stationary'
VIZ_ERR_THRESH = 50.0  # m — dense_chain mean centroid L2 above this triggers a plot


# ── Geometry helpers ──────────────────────────────────────────────────────────

def build_gt_contours(pos_xy, heading, width, length):
    """
    Compute canonical bounding box polygons for N agents over T frames.

    Args:
        pos_xy:  [N, T, 2]  x/y world positions
        heading: [N, T]     world headings (radians)
        width:   float      canonical bbox width
        length:  float      canonical bbox length

    Returns:
        [N, T, 4, 2]  four corner points per agent per frame
    """
    N, T, _ = pos_xy.shape
    contours = cal_polygon_contour(
        pos_xy[:, :, 0].ravel(),
        pos_xy[:, :, 1].ravel(),
        heading.ravel(),
        width, length,
    )                             # [N*T, 4, 2]
    return contours.reshape(N, T, 4, 2)


def decode_tokens(token_all, token_idx, anchor_pos, anchor_heading):
    """
    Decode token indices into world-frame bounding-box polygon sequences.

    The rotation convention matches agent_decoder.py inference:
        world_corners = local_corners @ [[cos θ,  sin θ],
                                         [-sin θ, cos θ]]  + anchor_pos

    Args:
        token_all:      [2048, 6, 4, 2]  canonical local-frame sequences
        token_idx:      [N, T_tok]       integer token indices ∈ [0, 2047]
        anchor_pos:     [N, T_tok, 2]    world position of each window's anchor
        anchor_heading: [N, T_tok]       world heading of each window's anchor

    Returns:
        [N, T_tok, 6, 4, 2]  world-frame polygon sequences
                              axis-2 indexes the 6 frames within the window
    """
    N, T_tok = token_idx.shape

    local_seqs = token_all[token_idx]          # [N, T_tok, 6, 4, 2]

    cos_h = np.cos(anchor_heading)             # [N, T_tok]
    sin_h = np.sin(anchor_heading)
    rot = np.zeros((N, T_tok, 2, 2))
    rot[..., 0, 0] =  cos_h
    rot[..., 0, 1] =  sin_h
    rot[..., 1, 0] = -sin_h
    rot[..., 1, 1] =  cos_h

    # [N, T_tok, 24, 2] einsum [N, T_tok, 2, 2] → [N, T_tok, 24, 2]
    local_flat = local_seqs.reshape(N, T_tok, -1, 2)
    world_flat = np.einsum('...ki,...ij->...kj', local_flat, rot)
    world_seqs = world_flat.reshape(N, T_tok, 6, 4, 2)

    world_seqs = world_seqs + anchor_pos[:, :, None, None, :]   # translate
    return world_seqs


# ── Metrics ───────────────────────────────────────────────────────────────────

def corner_l2(pred, gt):
    """Mean L2 distance over 4 corners.  pred/gt: [..., 4, 2] → [...]"""
    return np.sqrt(((pred - gt) ** 2).sum(-1)).mean(-1)


def centroid_l2(pred, gt):
    """L2 distance between polygon centroids.  pred/gt: [..., 4, 2] → [...]"""
    return np.sqrt(((pred.mean(-2) - gt.mean(-2)) ** 2).sum(-1))


def heading_err(pred, gt):
    """
    Absolute wrapped heading error in radians.  pred/gt: [..., 4, 2] → [...]
    Heading is derived from the left_front – left_back corner vector,
    consistent with how token_heading is computed in TokenProcessor.
    """
    def _heading(c):
        d = c[..., 0, :] - c[..., 3, :]      # left_front - left_back
        return np.arctan2(d[..., 1], d[..., 0])
    diff = _heading(pred) - _heading(gt)
    return np.abs((diff + np.pi) % (2 * np.pi) - np.pi)


# ── Visualization ─────────────────────────────────────────────────────────────

def _save_agent_plot(pos_gt, fvalid, world_c_agent, vmask_agent, anchor_ok_agent,
                     win_start, shift, T_full, mean_err,
                     scenario_id, cat, global_idx, viz_dir):
    """
    Save a scatter plot comparing GT positions and dense_chain reconstructed
    centroids for a single agent.

    Args:
        pos_gt:        [T_full, 2]   world-frame GT positions
        fvalid:        [T_full]      per-frame GT validity
        world_c_agent: [T_tok, 6, 4, 2]  decoded dense_chain world corners
        vmask_agent:   [T_tok]       per-token validity
        win_start:     [T_tok]       frame index of each token window start
        shift:         int           frames per token window
        T_full:        int           total trajectory length
        mean_err:      float         mean centroid L2 triggering this plot
        scenario_id:   str
        cat:           str           'veh' / 'ped' / 'cyc'
        global_idx:    int           agent index in the full scenario array
        viz_dir:       str / Path
    """
    import matplotlib.pyplot as plt

    # GT: positions at valid, non-fill frames
    gt_valid_mask = fvalid & ~np.all(pos_gt == 0, axis=-1)  # [T_full]
    gt_pts = pos_gt[gt_valid_mask]                           # [M, 2]

    # Reconstruction: centroid of each decoded frame (skip frame-0 anchor)
    recon_pts = []
    for t in range(world_c_agent.shape[0]):
        if not (vmask_agent[t] and anchor_ok_agent[t]):
            continue
        for f in range(1, shift + 1):
            frame_idx = min(win_start[t] + f, T_full - 1)
            if not (fvalid[frame_idx] and not np.all(pos_gt[frame_idx] == 0)):
                continue
            centroid = world_c_agent[t, f, :, :].mean(axis=0)  # [2]
            recon_pts.append(centroid)

    fig, ax = plt.subplots(figsize=(9, 7))

    if len(gt_pts):
        ax.scatter(gt_pts[:, 0], gt_pts[:, 1],
                   c='steelblue', s=12, label='GT', alpha=0.8, zorder=3)
        if len(gt_pts) > 1:
            ax.plot(gt_pts[:, 0], gt_pts[:, 1],
                    color='steelblue', lw=0.8, alpha=0.4)

    if recon_pts:
        recon = np.array(recon_pts)                             # [K, 2]
        ax.scatter(recon[:, 0], recon[:, 1],
                   c='tomato', s=12, label='Dense chain', alpha=0.8, zorder=3)
        if len(recon) > 1:
            ax.plot(recon[:, 0], recon[:, 1],
                    color='tomato', lw=0.8, alpha=0.4)

    ax.set_aspect('equal')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_title(
        f'{scenario_id}  |  {cat} agent {global_idx}'
        f'  |  mean centroid L2 = {mean_err:.1f} m',
        fontsize=10,
    )

    Path(viz_dir).mkdir(parents=True, exist_ok=True)
    fname = f'{scenario_id}_{cat}_{global_idx:03d}.png'
    fig.savefig(Path(viz_dir) / fname, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'  [viz] {fname}  (mean L2 = {mean_err:.1f} m)')


# ── Per-scenario evaluation ───────────────────────────────────────────────────

def evaluate_scenario(data, processor, results, viz_dir=None, scenario_id='scene'):
    """
    Run all three reconstruction modes on one scenario and accumulate metrics.

    Args:
        data:      HeteroData dict returned by TokenProcessor.preprocess()
        processor: TokenProcessor instance (used for shift and token_all lookup)
        results:   defaultdict(list) accumulating scalar metric values
    """
    token_idx = data['agent']['token_idx'].numpy() # [N, T_tok]
    token_contour = data['agent']['token_contour'].numpy() # [N, T_tok, 4, 2]
    token_pos = data['agent']['token_pos'].numpy() # [N, T_tok, 2]
    token_heading = data['agent']['token_heading'].numpy() # [N, T_tok]
    valid_mask = data['agent']['agent_valid_mask'].numpy() # [N, T_tok] — valid at window start+end only
    frame_valid = data['agent']['valid_mask'].numpy()      # [N, T_full] — per-frame validity

    pos = data['agent']['position'].numpy()[:, :, :2] # [N, T_full, 2]
    heading = data['agent']['heading'].numpy() # [N, T_full]
    atype = data['agent']['type'].numpy() # [N]

    shift = processor.shift
    T_full = pos.shape[1]
    T_tok = token_idx.shape[1]

    # Window boundary frame indices; clip to valid range
    win_start = np.arange(T_tok) * shift # [T_tok]
    win_end = np.minimum(win_start + shift, T_full - 1) # [T_tok]

    for type_id, cat in TYPE_TO_CAT.items():
        cat_mask = atype == type_id
        if not cat_mask.any():
            continue

        width, length = CAT_DIMS[cat]
        token_all_cat = processor.trajectory_token_all[cat]     # [2048, 6, 4, 2]

        pos_c = pos[cat_mask] # [n, T_full, 2]
        heading_c = heading[cat_mask] # [n, T_full]
        tidx_c = token_idx[cat_mask] # [n, T_tok]
        tc_c = token_contour[cat_mask] # [n, T_tok, 4, 2]
        tpos_c = token_pos[cat_mask] # [n, T_tok, 2]
        thead_c = token_heading[cat_mask] # [n, T_tok]
        vmask_c = valid_mask[cat_mask]           # [n, T_tok]
        fvalid_c = frame_valid[cat_mask]         # [n, T_full] — per-frame GT validity
        n = int(cat_mask.sum())

        # GT canonical bounding boxes at every frame: [n, T_full, 4, 2]
        gt_all = build_gt_contours(pos_c, heading_c, width, length)

        # Per-window speed (m/s) for motion stratification: [n, T_tok]
        disp  = pos_c[:, win_end, :] - pos_c[:, win_start, :]    # [n, T_tok, 2]
        speed = np.sqrt((disp ** 2).sum(-1)) / (shift * 0.1)     # [n, T_tok]
        moving = speed > MOVING_THRESH                             # [n, T_tok]

        # ── Endpoint eval ────────────────────────────────────────────────────
        # token_contour is the matched token's endpoint polygon placed in world
        # frame — compare directly against the GT bbox at the destination frame.
        gt_ep = gt_all[:, win_end, :, :]  # [n, T_tok, 4, 2]
        pred_ep = tc_c # [n, T_tok, 4, 2]

        ep_corner = corner_l2(pred_ep, gt_ep)
        ep_centroid = centroid_l2(pred_ep, gt_ep)
        ep_heading = heading_err(pred_ep, gt_ep)

        for mov_flag, motion_str in [(True, 'moving'), (False, 'stationary')]:
            sel = vmask_c & (moving == mov_flag)
            if not sel.any():
                continue
            key = f'{cat}.{motion_str}.endpoint'
            results[key + '.corner_l2'].extend(ep_corner[sel].tolist())
            results[key + '.centroid_l2'].extend(ep_centroid[sel].tolist())
            results[key + '.heading_err_deg'].extend(np.degrees(ep_heading[sel]).tolist())

        # ── Dense eval (independent) ─────────────────────────────────────────
        # Anchor each window at the actual trajectory pose at win_start.
        # Measures single-window quantisation error with no cross-window
        # error accumulation.
        anc_pos_i  = pos_c[:, win_start, :]    # [n, T_tok, 2]
        anc_head_i = heading_c[:, win_start]   # [n, T_tok]
        anc_ok_i   = ~np.all(anc_pos_i == 0, axis=-1)  # [n, T_tok] exclude fill anchors

        world_i = decode_tokens(token_all_cat, tidx_c, anc_pos_i, anc_head_i)
        # world_i: [n, T_tok, 6, 4, 2]

        # Frame 0 of each token is the anchor itself; compare frames 1..shift
        for f in range(1, shift + 1):
            frames = np.minimum(win_start + f, T_full - 1)   # [T_tok]
            gt_frame_valid = (fvalid_c[:, frames]
                              & ~np.all(pos_c[:, frames, :] == 0, axis=-1))  # [n, T_tok]
            gt_f   = gt_all[:, frames, :, :]                  # [n, T_tok, 4, 2]
            pred_f = world_i[:, :, f, :, :]                   # [n, T_tok, 4, 2]

            di_corner   = corner_l2(pred_f, gt_f)
            di_centroid = centroid_l2(pred_f, gt_f)
            di_heading  = heading_err(pred_f, gt_f)

            for mov_flag, motion_str in [(True, 'moving'), (False, 'stationary')]:
                sel = vmask_c & anc_ok_i & gt_frame_valid & (moving == mov_flag)
                if not sel.any():
                    continue
                key = f'{cat}.{motion_str}.dense_indep'
                results[key + '.corner_l2'].extend(di_corner[sel].tolist())
                results[key + '.centroid_l2'].extend(di_centroid[sel].tolist())
                results[key + '.heading_err_deg'].extend(np.degrees(di_heading[sel]).tolist())

        # ── Dense eval (chained) ─────────────────────────────────────────────
        # Anchor window t at the EXIT POSE of token t-1, exactly mirroring
        # match_token and the inference rollout.  Error accumulates across
        # windows, reflecting real reconstruction fidelity during simulation.
        # Mirror match_token's hybrid anchoring:
        #   end of each iteration: prev_pos = pos[:, i].clone()
        #                          prev_pos[valid at window_start] = token_exit
        # So anchor for window t is:
        #   token exit of window t-1  if agent was valid at (t-1)*shift
        #   GT pos at t*shift         otherwise (GT reset, as in match_token)
        anc_pos_c  = np.empty((n, T_tok, 2))
        anc_head_c = np.empty((n, T_tok))
        anc_pos_c[:, 0, :]  = pos_c[:, 0, :]
        anc_head_c[:, 0]    = heading_c[:, 0]
        for t in range(1, T_tok):
            prev_start = (t - 1) * shift
            curr_start = min(t * shift, T_full - 1)
            was_valid  = fvalid_c[:, prev_start]          # [n] bool
            anc_pos_c[:, t, :]  = np.where(was_valid[:, None],
                                            tpos_c[:, t - 1, :],
                                            pos_c[:, curr_start, :])
            anc_head_c[:, t]    = np.where(was_valid,
                                            thead_c[:, t - 1],
                                            heading_c[:, curr_start])

        # Anchor validity: exclude windows where the anchor itself is a fill (0,0)
        # position — these arise when the agent hasn't appeared yet and
        # pos_c[:, t*shift, :] is still the dataset fill value.
        anchor_ok = ~np.all(anc_pos_c == 0, axis=-1)  # [n, T_tok]

        world_c = decode_tokens(token_all_cat, tidx_c, anc_pos_c, anc_head_c)

        for f in range(1, shift + 1):
            frames = np.minimum(win_start + f, T_full - 1)
            gt_frame_valid = (fvalid_c[:, frames]
                              & ~np.all(pos_c[:, frames, :] == 0, axis=-1))  # [n, T_tok]
            gt_f = gt_all[:, frames, :, :]
            pred_f = world_c[:, :, f, :, :]

            dc_corner = corner_l2(pred_f, gt_f)
            dc_centroid = centroid_l2(pred_f, gt_f)
            dc_heading = heading_err(pred_f, gt_f)

            for mov_flag, motion_str in [(True, 'moving'), (False, 'stationary')]:
                sel = vmask_c & anchor_ok & gt_frame_valid & (moving == mov_flag)
                if not sel.any():
                    continue
                key = f'{cat}.{motion_str}.dense_chain'
                results[key + '.corner_l2'].extend(dc_corner[sel].tolist())
                results[key + '.centroid_l2'].extend(dc_centroid[sel].tolist())
                results[key + '.heading_err_deg'].extend(np.degrees(dc_heading[sel]).tolist())

        # ── Visualization of high-error agents ───────────────────────────────
        if viz_dir is not None:
            global_indices = np.where(cat_mask)[0]  # map local→scenario agent idx
            recon_cents = world_c.mean(axis=-2)      # [n, T_tok, 6, 2]
            for ai in range(n):
                agent_errs = []
                for f in range(1, shift + 1):
                    frames = np.minimum(win_start + f, T_full - 1)  # [T_tok]
                    gfv = (fvalid_c[ai, frames]
                           & ~np.all(pos_c[ai, frames, :] == 0, axis=-1))
                    sel = vmask_c[ai] & anchor_ok[ai] & gfv
                    if not sel.any():
                        continue
                    pred_c = recon_cents[ai, :, f, :]   # [T_tok, 2]
                    gt_c   = pos_c[ai, frames, :]        # [T_tok, 2]
                    errs   = np.sqrt(((pred_c - gt_c) ** 2).sum(-1))  # [T_tok]
                    agent_errs.extend(errs[sel].tolist())

                if agent_errs and np.mean(agent_errs) > VIZ_ERR_THRESH:
                    _save_agent_plot(
                        pos_gt=pos_c[ai],
                        fvalid=fvalid_c[ai],
                        world_c_agent=world_c[ai],
                        vmask_agent=vmask_c[ai],
                        anchor_ok_agent=anchor_ok[ai],
                        win_start=win_start,
                        shift=shift,
                        T_full=T_full,
                        mean_err=np.mean(agent_errs),
                        scenario_id=scenario_id,
                        cat=cat,
                        global_idx=int(global_indices[ai]),
                        viz_dir=viz_dir,
                    )


# ── Output ────────────────────────────────────────────────────────────────────

def print_table(results):
    cats  = ['veh', 'ped', 'cyc']
    motions = ['moving', 'stationary']
    modes = ['endpoint', 'dense_indep', 'dense_chain']
    metrics = ['corner_l2', 'centroid_l2', 'heading_err_deg']
    headers = ['corner L2 (m)', 'centroid L2 (m)', 'heading err (°)']

    lw = 40    # label column width
    cw = 26    # metric column width (wider to fit mean/median)
    sep = '─' * (lw + cw * len(metrics))

    print()
    print(sep)
    print(f"{'':>{lw}}" + ''.join(f'{h:>{cw}}' for h in headers))
    print(f"{'  mean (median)':>{lw}}" + ''.join(f'{"mean (median)":>{cw}}' for _ in metrics))
    print(sep)

    for cat in cats:
        for motion in motions:
            for mode in modes:
                row_key = f'{cat}.{motion}.{mode}'
                label   = f'{cat:3s}  {motion:10s}  {mode}'
                cols = []
                for m in metrics:
                    v = results.get(f'{row_key}.{m}', [])
                    if v:
                        arr = np.array(v)
                        cols.append(f'{arr.mean():.3f} ({np.median(arr):.3f})')
                    else:
                        cols.append('—')
                print(f'{label:<{lw}}' + ''.join(f'{c:>{cw}}' for c in cols))
            print()
        print(sep)
    print()


def save_csv(results, path):
    import csv
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['key', 'mean', 'std', 'n'])
        for k, v in sorted(results.items()):
            writer.writerow([k, f'{np.mean(v):.6f}', f'{np.std(v):.6f}', len(v)])
    print(f'Results saved to {path}')


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Evaluate SMART action tokenizer reconstruction quality.')
    parser.add_argument(
        '--data_dir', type=str, required=True,
        help='Directory of preprocessed scenario .pkl files '
             '(output of data_preprocess.py).')
    parser.add_argument(
        '--num_scenarios', type=int, default=None,
        help='Max number of scenarios to process (default: all).')
    parser.add_argument(
        '--output_csv', type=str, default=None,
        help='Optional path to write per-key mean/std/n as CSV.')
    parser.add_argument(
        '--viz_dir', type=str, default=None,
        help='Directory to save trajectory plots for agents with dense_chain '
             f'mean centroid L2 > {VIZ_ERR_THRESH} m.')
    args = parser.parse_args()

    processor = TokenProcessor(token_size=2048)
    processor.training = False
    processor.noise    = False

    pkl_files = sorted(Path(args.data_dir).glob('*.pkl'))
    if not pkl_files:
        raise FileNotFoundError(f'No .pkl files found in {args.data_dir}')
    if args.num_scenarios is not None:
        pkl_files = pkl_files[:args.num_scenarios]

    print(f'Evaluating {len(pkl_files)} scenarios from {args.data_dir} ...')

    results = defaultdict(list)
    n_ok, n_skip = 0, 0

    for i, fpath in enumerate(pkl_files):
        try:
            with open(fpath, 'rb') as f:
                data = pickle.load(f)
            data = processor.preprocess(data)
            scenario_id = str(data.get('scenario_id', fpath.stem))
            evaluate_scenario(data, processor, results,
                              viz_dir=args.viz_dir, scenario_id=scenario_id)
            n_ok += 1
        except Exception as e:
            print(f'  [warn] skipped {fpath.name}: {e}')
            n_skip += 1

        if (i + 1) % 100 == 0 or (i + 1) == len(pkl_files):
            print(f'  {i + 1}/{len(pkl_files)} processed  '
                  f'({n_ok} ok, {n_skip} skipped)')

    print(f'\nFinished. {n_ok} scenarios contributed to results.\n')
    print_table(results)

    if args.output_csv:
        save_csv(results, args.output_csv)


if __name__ == '__main__':
    main()
