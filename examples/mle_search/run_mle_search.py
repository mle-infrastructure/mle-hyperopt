def main(config):
    """Optimum: lrate=0.2, batch_size=4, arch='conv'."""
    f1 = (
        (config["lrate"] - 0.2) ** 2
        + ((config["batch_size"] - 4) / 4) ** 2
        + (0 if config["arch"]["sub_arch"] == "conv" else 0.2)
    )
    return f1
