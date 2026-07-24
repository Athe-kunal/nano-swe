# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from typing import Any, Dict

from loguru import logger as _logger
from omegaconf import DictConfig, OmegaConf


def init_logger(name: str):
    return _logger.bind(name=name)


class WandbLogger:
    """Handle wandb setup and training-time logging."""

    def __init__(self, args: DictConfig) -> None:
        import wandb

        if not wandb.api.api_key:
            wandb.login(key=args.logger.wandb.key)
        wandb.init(
            entity=args.logger.wandb.org,
            project=args.logger.wandb.project,
            group=args.logger.wandb.group,
            name=args.logger.wandb.run_name,
            config=OmegaConf.to_container(args, resolve=True),
            reinit=True,
        )

        wandb.define_metric("train/global_step")
        wandb.define_metric("train/*", step_metric="train/global_step", step_sync=True)
        wandb.define_metric("eval/epoch")
        wandb.define_metric("eval/*", step_metric="eval/epoch", step_sync=True)
        self.handle = wandb
        self.samples_table = wandb.Table(columns=["global_step", "text", "reward"])

    def log_train(self, global_step: int, logs_dict: Dict[str, Any]) -> None:
        logs_dict = dict(logs_dict)

        generated_samples = logs_dict.pop("generated_samples", None)
        if generated_samples:
            # https://github.com/wandb/wandb/issues/2981#issuecomment-1997445737
            new_table = self.handle.Table(columns=self.samples_table.columns, data=self.samples_table.data)
            new_table.add_data(global_step, *generated_samples)
            self.samples_table = new_table
            self.handle.log({"train/generated_samples": new_table})

        metrics = {k: v for k, v in logs_dict.items() if v is not None}
        logs = {"train/%s" % k: v for k, v in {**metrics, "global_step": global_step}.items()}
        self.handle.log(logs)

    def log_eval(self, global_step: int, logs_dict: Dict[str, Any]) -> None:
        logs_dict = dict(logs_dict)

        metrics = {k: v for k, v in logs_dict.items() if v is not None}
        logs = {"eval/%s" % k: v for k, v in {**metrics, "global_step": global_step}.items()}
        self.handle.log(logs)

    def close(self) -> None:
        self.handle.finish()


class TensorboardLogger:
    """Handle tensorboard setup and training-time logging."""

    def __init__(self, args: DictConfig) -> None:
        from torch.utils.tensorboard import SummaryWriter

        os.makedirs(args.logger.tensorboard_dir, exist_ok=True)
        log_dir = os.path.join(args.logger.tensorboard_dir, args.logger.wandb.run_name)
        self.writer = SummaryWriter(log_dir=log_dir)

    def log_train(self, global_step: int, logs_dict: Dict[str, Any]) -> None:
        generated_samples = logs_dict.get("generated_samples")
        for k, v in logs_dict.items():
            if k == "generated_samples" and v is not None:
                text, reward = generated_samples
                formatted_text = f"Sample:\n{text}\n\nReward: {reward:.4f}"
                self.writer.add_text("train/generated_samples", formatted_text, global_step)
            elif v is not None:
                self.writer.add_scalar(f"train/{k}", v, global_step)

    def log_eval(self, global_step: int, logs_dict: Dict[str, Any]) -> None:
        for k, v in logs_dict.items():
            if v is not None:
                self.writer.add_scalar(f"eval/{k}", v, global_step)

    def close(self) -> None:
        self.writer.close()