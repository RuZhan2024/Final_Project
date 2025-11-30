#!/usr/bin/env python3
import json, argparse
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reports', required=True)
    ap.add_argument('--title', default='FA/24h vs Recall')
    ap.add_argument('--subtitle', default='')
    ap.add_argument('--out_fig', required=True)
    args = ap.parse_args()

    with open(args.reports, 'r') as f:
        rep = json.load(f)

    # Accept either a dict with 'ops' or a plain mapping of op_name -> dict
    if isinstance(rep, dict) and 'ops' in rep and isinstance(rep['ops'], dict):
        items = list(rep['ops'].items())
    elif isinstance(rep, dict):
        # fall back: assume the whole dict maps names -> metrics
        items = list(rep.items())
    else:
        raise SystemExit("Unexpected report format: expected a dict (optionally with 'ops').")

    # Collect metrics
    points = []
    fa_present = False
    for name, d in items:
        if not isinstance(d, dict):
            continue
        thr = d.get('thr')
        rec = d.get('recall')
        pre = d.get('precision')
        fa  = d.get('fa24h')  # may be absent
        if rec is not None:
            points.append(dict(name=name, thr=thr, recall=rec, precision=pre, fa24h=fa))
            if fa is not None:
                fa_present = True

    if not points:
        raise SystemExit("No usable metrics found in report (need at least recall values).")

    # Slightly larger figure for readability
    plt.figure(figsize=(6, 4))

    def annotate_points(xs, ys, labels, title, xlabel, ylabel):
        ax = plt.gca()
        ax.scatter(xs, ys)

        n = len(xs)
        # Sort by x so labels are staggered in a consistent order
        order = sorted(range(n), key=lambda i: xs[i])

        # Base diagonal offset (in screen points) and step between labels
        base = 6       # starting offset
        step = 10      # extra offset per label
        # We’ll move labels along a 45° line: dx > 0, dy < 0 (top-left → bottom-right)
        # For rank r: offset_r = base + r*step, then (dx, dy) = (offset_r, -offset_r)

        for rank, i in enumerate(order):
            x, y, label = xs[i], ys[i], labels[i]
            offset = base + rank * step
            dx = offset
            dy = -offset

            ax.annotate(
                label,
                (x, y),
                xytext=(dx, dy),
                textcoords='offset points',
                fontsize=8,
                rotation=-45,              # top-left → bottom-right
                rotation_mode='anchor',
                bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.8),
                arrowprops=dict(arrowstyle='-', lw=0.5, alpha=0.7)
            )

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if args.subtitle:
            ax.set_title(title + "\n" + args.subtitle)
        else:
            ax.set_title(title)

        ax.grid(True, linestyle='--', alpha=0.3)

    if fa_present:
        # Plot FA/24h vs Recall
        xs, ys, labels = [], [], []
        for p in points:
            if p['fa24h'] is None:
                continue
            xs.append(p['fa24h'])
            ys.append(p['recall'])
            lab_thr = f"{p['thr']:.2f}" if isinstance(p['thr'], (int, float)) else str(p['thr'])
            labels.append(f"{p['name']} (thr={lab_thr})")

        if not xs:
            raise SystemExit("Report indicates fa24h exists, but no points had it populated.")

        annotate_points(
            xs,
            ys,
            labels,
            title=args.title,
            xlabel='FA/24h (Lower is better)',
            ylabel='Recall (Higher is better)',
        )
        # lower FA to the right
        plt.gca().invert_xaxis()
    else:
        # Fallback: Precision vs Recall
        xs, ys, labels = [], [], []
        for p in points:
            if p['precision'] is None:
                continue
            xs.append(p['precision'])
            ys.append(p['recall'])
            lab_thr = f"{p['thr']:.2f}" if isinstance(p['thr'], (int, float)) else str(p['thr'])
            labels.append(f"{p['name']} (thr={lab_thr})")

        if not xs:
            raise SystemExit("Report has neither 'fa24h' nor precision/recall per op.")

        annotate_points(
            xs,
            ys,
            labels,
            title='Precision vs Recall (FA/24h not in report)',
            xlabel='Precision',
            ylabel='Recall',
        )

    plt.tight_layout()
    plt.savefig(args.out_fig, dpi=180)
    print('[fig]', args.out_fig)

if __name__ == '__main__':
    main()
