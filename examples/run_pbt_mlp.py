from mle_hyperopt import PBTSearch
from mle_scheduler import MLEQueue


def main():
    """Run Population-Based Training"""
    strategy = PBTSearch(
        real={"lrate": {"begin": 1e-08, "end": 1e-06, "prior": "uniform"}},
        search_config={
            "exploit": {"strategy": "truncation", "selection_percent": 0.2},
            "explore": {"strategy": "perturbation", "perturb_coeffs": [0.8, 1.2]},
            "steps_until_ready": 4,
            "num_workers": 5,
        },
        seed_id=42,
        verbose=True,
    )

    num_pbt_steps = 20
    # Run a queue of successive halving batch jobs
    for pbt_iter in range(num_pbt_steps):
        configs, config_fnames = strategy.ask(store=True)
        queue = MLEQueue(
            resource_to_run="local",
            job_filename="train_mlp.py",
            config_filenames=config_fnames,
            experiment_dir="logs_pbt",
            max_running_jobs=5,
            automerge_configs=True,
        )
        queue.run()

        scores = [queue.log[r].stats.loss.mean[-1] for r in queue.mle_run_ids]
        ckpts = [queue.log[r].meta.model_ckpt for r in queue.mle_run_ids]
        strategy.tell(configs, scores, ckpts, save=True)


if __name__ == "__main__":
    main()
