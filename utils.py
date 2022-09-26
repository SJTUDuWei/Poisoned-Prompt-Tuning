import os
import sys
import logging


def set_logging(output_dir, log_file_name):
    log_format = '%(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format)
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, log_file_name)
    if os.path.exists(log_file):
        os.remove(log_file)
    fh = logging.FileHandler(log_file)
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)


def convergence(best_score, score_traces, max_steps, eval_every_steps):
    thres99 = 0.99 * best_score
    thres98 = 0.98 * best_score
    thres100 = best_score
    step100 = step99 = step98 = max_steps
    for val_time, score in enumerate(score_traces):
        if score >= thres98:
            step98 = min((val_time+1) * eval_every_steps, step98)
            if score >= thres99:
                step99 = min((val_time+1) * eval_every_steps, step99)
                if score >= thres100:
                    step100 = min((val_time+1) * eval_every_steps, step100)
    return step98, step99, step100


