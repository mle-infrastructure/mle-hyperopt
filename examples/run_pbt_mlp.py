from mle_hyperopt import PBTSearch
from mle_scheduler import MLEQueue


def main():
    """Run Population-Based Training"""
    strategy = PBTSearch(
        real={"lrate": {"begin": 1e-08, "end": 1e-06, "prior": "uniform"}},
        search_config={"min_budget": 1, "num_arms": 20, "halving_coeff": 2},
        seed_id=42,
        verbose=True,
    )
    print(strategy.num_batches, strategy.evals_per_batch, strategy.iters_per_batch)

    # Run a queue of successive halving batch jobs
    for sh_iter in range(strategy.num_batches):
        configs, config_fnames = strategy.ask(store=True)
        if type(config_fnames) == str:
            config_fnames = [config_fnames]
        queue = MLEQueue(
            resource_to_run="local",
            job_filename="train_mlp.py",
            config_filenames=config_fnames,
            experiment_dir="logs_sh",
            max_running_jobs=4,
            automerge_configs=True,
        )
        queue.run()
        if sh_iter + 1 != strategy.num_batches:
            scores = [queue.log[r].stats.loss.mean[-1] for r in queue.mle_run_ids]
            ckpts = [queue.log[r].meta.model_ckpt for r in queue.mle_run_ids]
        else:
            scores = [queue.log.stats.loss.mean[-1]]
            ckpts = [queue.log.meta.model_ckpt]
        strategy.tell(configs, scores, ckpts)


if __name__ == "__main__":
    main()
