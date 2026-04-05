#!/usr/bin/env python
"""
RaCo training script with eval and TensorBoard logging.
Follows glue-factory training pattern.
"""

import argparse
from datetime import datetime
from pathlib import Path
import torch
from loguru import logger
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter

from raco.datasets import get_dataset
from raco.models import get_model
from raco.utils.tensorboard_vis import create_scene_logger
from raco.evaluation import run_eval
from raco.trainer import StageTrainer, save_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--conf", type=str, default="configs/default.yaml")
    parser.add_argument("-s", "--stage", type=str, default="detector",
                        choices=["detector", "ranker", "covariance", "all"])
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--eval_only", action="store_true", help="Run eval only")
    args = parser.parse_args()

    # Load config
    conf = OmegaConf.load(args.conf)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Create output dir with timestamp subfolder
    base_output_dir = Path(conf.output.output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = base_output_dir / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    conf.output.output_dir = output_dir  # update
    logger.info(f"Output directory: {output_dir}")

    # Save config
    with open(output_dir / "config.yaml", "w") as f:
        OmegaConf.save(conf, f)

    # TensorBoard
    writer = SummaryWriter(output_dir / "tb_logs")

    # Create scene logger for tracking specific HPatches scenes
    scene_logger = create_scene_logger(writer)

    # Load model
    model = get_model(conf.model.name)(conf.model).to(device)

    # Compile model for faster training (PyTorch 2.0+)
    if conf.train.get("compile", False) and hasattr(torch, "compile"):
        logger.info("Compiling model with torch.compile()")
        model = torch.compile(model)

    if args.resume:
        logger.info(f"Loading checkpoint from {args.resume}")
        model.load_state_dict(torch.load(args.resume, map_location=device))

    # Load datasets
    train_dataset = get_dataset(conf.dataset.name)(conf.dataset)
    train_loader = train_dataset.get_data_loader("train")

    # Eval dataset (hpatches) if configured
    eval_loader = None
    if conf.train.get("eval_during_training", False):
        try:
            # HPatches uses data_dir, separate from oxford_paris data_root
            eval_conf = OmegaConf.create({
                "data_dir": conf.eval.get("data_root", "/mnt/e/datasets/hpatches-sequences-release"),
                "scene_type": "all",
                "batch_size": conf.eval.get("batch_size", 1),
                "num_workers": conf.eval.get("num_workers", 2),
                "max_scenes": conf.eval.get("max_scenes", None),
                "max_pairs_per_scene": conf.eval.get("max_pairs_per_scene", None),
            })
            eval_dataset = get_dataset("hpatches")(eval_conf)
            eval_loader = eval_dataset.get_data_loader("test", shuffle=False)
            logger.info(f"Eval dataset: {len(eval_loader.dataset)} pairs")
        except Exception as e:
            logger.warning(f"Could not load eval dataset: {e}")

    logger.info(f"Train dataset: {len(train_loader.dataset)} samples")
    logger.info(f"Model: {conf.model.name}")

    # Run eval only
    if args.eval_only and eval_loader is not None:
        run_eval(model, eval_loader, device, writer, 0, num_vis=10, scene_logger=scene_logger)
        writer.close()
        return

    # Train
    start_iter = 0
    if conf.train.stage == "all":
        stages = ["detector", "ranker", "covariance"]
        logger.info(f"Training all stages...")
    else:
        stages = [conf.train.stage]

    for stage in stages:
        trainer = StageTrainer(model, stage, conf, device, writer, scene_logger)
        start_iter = trainer.train(train_loader, eval_loader, start_iter)

        save_checkpoint(model, output_dir, stage, step=None)

    writer.close()
    logger.info("Training complete!")


if __name__ == "__main__":
    main()
