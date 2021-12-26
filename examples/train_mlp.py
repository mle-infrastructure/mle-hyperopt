import argparse
from mle_logging import MLELogger, load_config
import torch
import math


def main(experiment_dir: str, config_fname: str, seed_id: int):
    """Train 3rd order polynomial to approx sin(x) on -pi to pi. Following:
    github.com/pytorch/tutorials/blob/master/beginner_source/examples_nn/polynomial_nn.py"""
    train_config = load_config(config_fname, return_dotmap=True)
    log = MLELogger(
        experiment_dir=experiment_dir,
        config_fname=config_fname,
        seed_id=seed_id,
        time_to_track=["num_steps"],
        what_to_track=["loss"],
        model_type="torch",
        # verbose=True,
    )

    x = torch.linspace(-math.pi, math.pi, 2000)
    y = torch.sin(x)
    p = torch.tensor([1, 2, 3])
    xx = x.unsqueeze(-1).pow(p)

    model = torch.nn.Sequential(torch.nn.Linear(3, 1), torch.nn.Flatten(0, 1))

    # Reload model checkpoint if provided in config SH
    if "sh_ckpt" in train_config["extra"].keys():
        model.load_state_dict(torch.load(train_config.extra.sh_ckpt))
    elif "pbt_ckpt" in train_config["extra"].keys():
        model.load_state_dict(torch.load(train_config.extra.sh_ckpt))

    loss_fn = torch.nn.MSELoss(reduction="sum")

    for t in range(train_config.extra.sh_num_add_iters * 50):
        y_pred = model(xx)
        loss = loss_fn(y_pred, y)

        model.zero_grad()
        loss.backward()
        with torch.no_grad():
            for param in model.parameters():
                param -= float(train_config.params.lrate) * param.grad

        log.update(
            {"num_steps": t},
            {"loss": loss.item()},
            model=model,
            save=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Let's train a network.")
    parser.add_argument(
        "-exp_dir", "--experiment_dir", type=str, default="experiments/"
    )
    parser.add_argument(
        "-config", "--config_fname", type=str, default="base_config_1.yaml"
    )
    parser.add_argument("-seed", "--seed_id", type=int, default=1)
    args = vars(parser.parse_args())
    main(args["experiment_dir"], args["config_fname"], args["seed_id"])
