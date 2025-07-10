import argparse
import logging

import torch

import wandb
from core.distribution import GaussianVariable
from core.distribution.utils import from_copy, from_ivon
from core.metric import evaluate_metrics
from core.model import dnn_to_probnn
from core.risk import certify_risk
from core.split_strategy import PBPSplitStrategy
from core.training import train
from scripts.utils.config import get_wandb_name, load_config, setup_logging
from scripts.utils.factory import (
    BoundFactory,
    DataLoaderFactory,
    LossFactory,
    MetricFactory,
    ModelFactory,
    ObjectiveFactory,
)
from scripts.utils.training import train_ivon


def main(config: dict, config_path: str):
    if config["log_wandb"]:
        wandb.init(project="pbb_paper", config=config, name=get_wandb_name(config_path))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Device: {device}")
    # Losses
    logging.info(f"Selected losses: {config['factory']['losses']}")
    loss_factory = LossFactory()
    losses = {
        loss_name: loss_factory.create(loss_name)
        for loss_name in config["factory"]["losses"]
    }

    # Metrics
    logging.info(f"Select metrics: {config['factory']['metrics']}")
    metric_factory = MetricFactory()
    metrics = {
        metric_name: metric_factory.create(metric_name)
        for metric_name in config["factory"]["metrics"]
    }

    # Bound
    logging.info(f"Selected bounds: {config['factory']['bounds']}")
    bound_factory = BoundFactory()
    bounds = {
        bound_name: bound_factory.create(
            bound_name,
            bound_delta=config["bound"]["delta"],
            loss_delta=config["bound"]["delta_test"],
        )
        for bound_name in config["factory"]["bounds"]
    }

    # Data
    logging.info(f"Selected data loader: {config['factory']['data_loader']}")
    data_loader_factory = DataLoaderFactory()
    loader = data_loader_factory.create(
        config["factory"]["data_loader"]["name"],
        **config["factory"]["data_loader"]["params"],
    )

    strategy = PBPSplitStrategy(
        prior_type=config["split_strategy"]["prior_type"],
        train_percent=config["split_strategy"]["train_percent"],
        val_percent=config["split_strategy"]["val_percent"],
        prior_percent=config["split_strategy"]["prior_percent"],
        self_certified=config["split_strategy"]["self_certified"],
    )
    strategy.split(loader, split_config=config["split_config"])

    # Model
    logging.info(f"Select model: {config['factory']['model']['name']}")
    model_factory = ModelFactory()
    model = model_factory.create(
        config["factory"]["model"]["name"], **config["factory"]["model"]["params"]
    )

    torch.manual_seed(config["dist_init"]["seed"])
    model.to(device)

    # Training prior
    train_params = {
        "lr": config["prior"]["training"]["lr"],
        "momentum": config["prior"]["training"]["momentum"],
        "epochs": config["prior"]["training"]["epochs"],
        "seed": config["prior"]["training"]["seed"],
        "num_samples": strategy.prior_loader.batch_size * len(strategy.prior_loader),
        "train_samples": config["prior"]["training"]["train_samples"],
        "sigma": config["sigma"],
    }

    ivon = train_ivon(
        model=model,
        train_loader=strategy.prior_loader,
        val_loader=strategy.val_loader,
        parameters=train_params,
        device=device,
        wandb_params={"log_wandb": config["log_wandb"], "name_wandb": "Prior Train"},
    )

    posterior_prior = from_ivon(
        model, optimizer=ivon, distribution=GaussianVariable, requires_grad=False
    )
    posterior = from_copy(
        dist=posterior_prior, distribution=GaussianVariable, requires_grad=True
    )
    dnn_to_probnn(model, weight_dist=posterior, prior_weight_dist=posterior_prior)
    model.to(device)

    #  Train posterior
    train_params = {
        "lr": config["posterior"]["training"]["lr"],
        "momentum": config["posterior"]["training"]["momentum"],
        "epochs": config["posterior"]["training"]["epochs"],
        "seed": config["posterior"]["training"]["seed"],
        "num_samples": strategy.posterior_loader.batch_size
        * len(strategy.posterior_loader),
    }

    logging.info(
        f"Select objective: {config['factory']['posterior_objective']['name']}"
    )
    objective_factory = ObjectiveFactory()
    objective = objective_factory.create(
        config["factory"]["posterior_objective"]["name"],
        **config["factory"]["posterior_objective"]["params"],
    )

    train(
        model=model,
        posterior=posterior,
        prior=posterior_prior,
        objective=objective,
        train_loader=strategy.posterior_loader,
        val_loader=strategy.val_loader,
        parameters=train_params,
        device=device,
        wandb_params={
            "log_wandb": config["log_wandb"],
            "name_wandb": "Posterior Train",
        },
    )

    if strategy.test_loader is not None:
        _ = evaluate_metrics(
            model=model,
            metrics=metrics,
            test_loader=strategy.test_loader,
            num_samples_metric=config["mcsamples"],
            device=device,
            pmin=config["pmin"],
            wandb_params={
                "log_wandb": config["log_wandb"],
                "name_wandb": "Posterior Evaluation",
            },
        )

    _ = certify_risk(
        model=model,
        bounds=bounds,
        losses=losses,
        posterior=posterior,
        prior=posterior_prior,
        bound_loader=strategy.bound_loader,
        num_samples_loss=config["mcsamples"],
        device=device,
        pmin=config["pmin"],
        wandb_params={
            "log_wandb": config["log_wandb"],
            "name_wandb": "Posterior Bound",
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run script with a YAML config file")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the YAML config file"
    )
    args = parser.parse_args()
    config = load_config(args.config)
    setup_logging(args.config)
    main(config, args.config)
