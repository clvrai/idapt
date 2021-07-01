import cv2
import numpy as np
import os
import pickle
import time
import torch
import torch.optim as optim
import torchvision.transforms as transforms
import wandb
from PIL import Image
from itertools import cycle
from torch.optim.lr_scheduler import StepLR, LambdaLR

from training.agents.base_agent import BaseMultiStageAgent
from training.datasets import ReplayBuffer
from training.datasets import OfflineDataset
from training.networks.models import GOT_Model
from training.utils.info_dict import Info
from training.utils.logger import logger
from training.utils.pytorch import (
    to_tensor,
    sync_grads,
    count_parameters,
    sync_networks,
    optimizer_cuda,
    random_crop,
)


class GOTAgent(BaseMultiStageAgent):
    def __init__(
        self,
        config,
        source_ob_space,
        source_ac_space,
        target_ob_space,
        source_env_ob_space,
    ):
        super().__init__(config, source_ob_space, target_ob_space)

        assert source_ob_space == target_ob_space
        self._ob_space = source_ob_space
        self._source_ac_space = source_ac_space

        self._state_space = source_env_ob_space
        self._recon_coeff = config.recon_coeff

        ### Define Model ###
        self._encoder = GOT_Model(
            config, source_ob_space, target_ob_space, source_env_ob_space
        )
        self._network_cuda(config.device)
        self.output_dim = self._encoder.output_dim

        if "supervised_pretraining" in self._config.max_stage_steps.keys():
            ### Define Optimizers ###
            self._define_optimizers()

            ### Make Pretraining Dataset ###
            if "supervised_pretraining" in config.max_stage_steps.keys():
                source_dataset = OfflineDataset(
                    self._config.demo_subsample_interval,
                    self._source_ac_space,
                    use_low_level=self._config.demo_low_level,
                )

                source_eval_dataset = OfflineDataset(
                    self._config.demo_subsample_interval,
                    self._source_ac_space,
                    use_low_level=self._config.demo_low_level,
                )
                target_dataset = OfflineDataset(
                    self._config.demo_subsample_interval,
                    self._source_ac_space,
                    use_low_level=self._config.demo_low_level,
                )
                target_eval_dataset = OfflineDataset(
                    self._config.demo_subsample_interval,
                    self._source_ac_space,
                    use_low_level=self._config.demo_low_level,
                )
                self._datasets = {
                    "source": source_dataset,
                    "source_eval": source_eval_dataset,
                    "target": target_dataset,
                    "target_eval": target_eval_dataset,
                }

                if (
                    config.accumulate_data
                    or config.max_stage_steps["supervised_pretraining"] > 0
                ):
                    if self._config.target_demo_path:
                        self._datasets["target"].add_demos_from_path(
                            self._config.target_demo_path
                        )
                    if self._config.source_demo_path:
                        self._datasets["source"].add_demos_from_path(
                            self._config.source_demo_path
                        )
                    if self._config.target_eval_demo_path:
                        self._datasets["target_eval"].add_demos_from_path(
                            self._config.target_eval_demo_path
                        )
                    if self._config.source_eval_demo_path:
                        self._datasets["source_eval"].add_demos_from_path(
                            self._config.source_eval_demo_path
                        )

        self.set_encoder_requires_grad(False)
        self._log_creation()

    ### Data Gathering/Loading ###

    def store_episode(self, rollouts, env_type):
        demos = {
            "obs": rollouts["ob"] + [rollouts["ob_next"][-1]],
        }
        if "state" in rollouts.keys():  # "source" in env_type
            demos["state"] = rollouts["state"]
        if "qpos" in rollouts.keys() and "qvel" in rollouts.keys():
            demos["qpos"] = rollouts["qpos"]
            demos["qvel"] = rollouts["qvel"]
            

        self._datasets[env_type].add_demos(demos)

        if self._config.save_data:
            if "source" in env_type:
                env_name = self._config.source_env
            elif "target" in env_type:
                env_name = self._config.target_env
            print(
                "saving: ",
                env_type,
                env_name,
                self._config.source_env,
                self._config.target_env,
            )
            fname = "{}_{}_{}_transitions.pkl".format(
                env_name, self._config.run_prefix, len(demos["obs"]),
            )

            path = os.path.join(self._config.demo_dir, fname)
            logger.warn("[*] Generating demo: {}".format(path))
            with open(path, "wb") as f:
                pickle.dump(demos, f)

    def load_dataset(self):

        self._target_dataloader = torch.utils.data.DataLoader(
            self._datasets["target"],
            batch_size=self._config.GOT_batch_size,
            shuffle=True,
            drop_last=True,
        )
        self._source_dataloader = torch.utils.data.DataLoader(
            self._datasets["source"],
            batch_size=self._config.GOT_batch_size,
            shuffle=True,
            drop_last=True,
        )
        self._target_eval_dataloader = torch.utils.data.DataLoader(
            self._datasets["target_eval"],
            batch_size=self._config.GOT_batch_size,
            shuffle=True,
            drop_last=True,
        )
        self._source_eval_dataloader = torch.utils.data.DataLoader(
            self._datasets["source_eval"],
            batch_size=self._config.GOT_batch_size,
            shuffle=True,
            drop_last=True,
        )

    def clear_dataset(self):
        for key, item in self._datasets.items():
            item.clear()

    def act(self, ob, env_type, is_train=True, **kwargs):
        ### action should be decided by GATAgent
        raise NotImplementedError

    ### Training Functions ###

    def train(self, curr_step, grounding_step):
        self.set_encoder_requires_grad(True)

        train_info = Info()

        if curr_step == 1:
            self.load_dataset()

        if (
            self._config.include_recon
            and curr_step == self._config.source_encoder_train_epoch + 1
        ):
            print("Coping over Sim Encoder weights to target Encoder.")
            self._copy_target_network(
                self._encoder.target_encoder, self._encoder.source_encoder
            )
            self._encoder.set_source_networks_requires_grad(False)

        start = time.time()
        if (
            self._config.include_recon
            and curr_step <= self._config.source_encoder_train_epoch
        ):
            encoder_train_info = self._update_forward_encoder()

        else:
            encoder_train_info = self._update_GAN()

        encoder_train_info["encoder_pretraining_epoch"] = (
            curr_step
            + grounding_step * self._config.max_stage_steps["supervised_training"]
            + (grounding_step > 0)
            * self._config.max_stage_steps["supervised_pretraining"]
        )
        end = time.time()
        logger.info(
            "Encoder Pretraining Epoch: %f, Time: %f, Generator lr %f, Discriminator %f",
            curr_step,
            end - start,
            self._encoder_G_lr_scheduler.get_lr()[0],
            self._encoder_D_lr_scheduler.get_lr()[0],
        )

        if curr_step % 5 == 0:
            with torch.no_grad():
                if (
                    self._config.include_recon
                    and curr_step <= self._config.source_encoder_train_epoch
                ):
                    _test_info = self._update_forward_encoder(train=False)
                else:
                    _test_info = self._update_GAN(train=False)
            encoder_train_info.add(_test_info)

        train_info.add(encoder_train_info)
        self.set_encoder_requires_grad(False)

        return train_info

    def _update_forward_encoder(self, train=True):
        info = Info()
        _to_tensor = lambda x: to_tensor(x, self._config.device)
        losses = {"state_reconstruction": []}

        if train:
            source_dataloader = self._source_dataloader
            target_dataloader = self._target_dataloader
        else:
            source_dataloader = self._source_eval_dataloader
            target_dataloader = self._target_eval_dataloader

        num_batches = 0
        for source_transitions, target_transitions in zip(
            source_dataloader, cycle(target_dataloader)
        ):
            num_batches += 1

            ob_source = source_transitions["ob"].copy()

            ob_source = _to_tensor(ob_source)
            state_source = _to_tensor(source_transitions["state"])

            if train:
                loss = self._encoder.state_recon_loss(
                    ob_source,
                    state_source,
                    encoder_domain="source",
                    obs_domain="source",
                )

                self._encoder.zero_grad()
                loss.backward()
                sync_grads(self._encoder.source_encoder)
                sync_grads(self._encoder.predict)
                self._encoder_optim.step()

            else:
                with torch.no_grad():
                    loss = self._encoder.state_recon_loss(
                        ob_source,
                        state_source,
                        encoder_domain="source",
                        obs_domain="source",
                    )

            losses["state_reconstruction"].append(loss.detach().cpu().item())

        if train:
            self._encoder_lr_scheduler.step()

            for k, v in losses.items():
                if len(v) > 0:
                    info["encoder_{}_loss".format(k)] = np.mean(v)
                    print(
                        "[TRAIN] Encoder {} Loss: ".format(k),
                        info["encoder_{}_loss".format(k)],
                    )
        else:
            for k, v in losses.items():
                if len(v) > 0:
                    info["eval_encoder_{}_loss".format(k)] = np.mean(v)
                    print(
                        "[TEST] Encoder {} Loss: ".format(k),
                        info["eval_encoder_{}_loss".format(k)],
                    )

        return info

    def _update_GAN(self, train=True):
        info = Info()
        _to_tensor = lambda x: to_tensor(x, self._config.device)
        losses = {
            "generator": [],
            "discriminator": [],
            "GAN_state_reconstruction": [],
            "state_reconstruction": [],
            "GAN_feature_reconstruction": [],
            "gp": [],
        }
        losses["cycle_consistency"] = []
        losses["generator2"] = []
        losses["discriminator2"] = []
        real_outputs = []
        gen_outputs = []

        if train:
            source_dataloader = self._source_dataloader
            target_dataloader = self._target_dataloader
        else:
            source_dataloader = self._source_eval_dataloader
            target_dataloader = self._target_eval_dataloader

        num_batches = 0
        for source_transitions, target_transitions in zip(
            source_dataloader, cycle(target_dataloader)
        ):
            num_batches += 1
            ob_source = source_transitions["ob"].copy()

            ob_source = _to_tensor(ob_source)
            state_source = _to_tensor(source_transitions["state"])

            if isinstance(target_transitions["ob"], dict):
                ob_target = target_transitions["ob"].copy()
            else:
                ob_target = {"ob": target_transitions["ob"]}.copy()

            ob_target = _to_tensor(ob_target)
            ob_gen_target = self._encoder.generate(ob_source)

            ob_gen_source = self._encoder.generate(ob_target, target_domain="source")
            ob_rec_target = self._encoder.generate(
                ob_gen_source, target_domain="target"
            )
            ob_rec_source = self._encoder.generate(
                ob_gen_target, target_domain="source"
            )

            if train:

                ### Train Discriminator ###
                self._encoder.set_dis_networks_requires_grad(True)

                disc_ob_gen_target = ob_gen_target
                disc_ob_gen_source = ob_gen_source

                D_loss, real_output, gen_output = self._encoder.GAN_discriminator_loss(
                    ob_target, disc_ob_gen_target
                )

                (
                    D2_loss,
                    real_output2,
                    gen_output2,
                ) = self._encoder.GAN_discriminator_loss(
                    ob_source, disc_ob_gen_source, target_domain="source"
                )
                D_loss += D2_loss

                self._encoder.zero_grad()
                D_loss.backward()
                for key, net in self._encoder.D.items():
                    sync_grads(net)
                self._encoder_D_optim.step()

                ### Train Generator ###
                self._encoder.set_dis_networks_requires_grad(False)

                loss = 0

                G2_loss = self._encoder.GAN_generator_loss(
                    ob_gen_source, target_domain="source"
                )
                cycle_loss = self._encoder.GAN_cycle_loss(
                    ob_target, ob_rec_target
                ) + self._encoder.GAN_cycle_loss(ob_source, ob_rec_source)
                loss += G2_loss + self._config.cycle_coeff * cycle_loss

                if self._config.include_recon:
                    if self._config.recon_state:
                        recon_loss = self._encoder.state_recon_loss(
                            ob_gen_target,
                            state_source,
                            encoder_domain="target",
                            obs_domain="target",
                        )
                        loss += self._recon_coeff * recon_loss
                    else:
                        feat_recon_loss = self._encoder.feature_recon_loss(
                            ob_source, ob_gen_target
                        )
                        loss += self._recon_coeff * feat_recon_loss

                G_loss = self._encoder.GAN_generator_loss(ob_gen_target)
                loss += G_loss

                self._encoder.zero_grad()
                loss.backward()
                for key, net in self._encoder.G.items():
                    sync_grads(net)
                if self._config.include_recon:
                    sync_grads(self._encoder.target_encoder)
                self._encoder_G_optim.step()

            else:
                with torch.no_grad():
                    ob_gen_target = self._encoder.generate(ob_source)
                    if self._config.include_recon:
                        if not self._config.recon_state:
                            feat_recon_loss = self._encoder.feature_recon_loss(
                                ob_source, ob_gen_target
                            )
                        state_target = _to_tensor(target_transitions["state"])
                        recon_loss = self._encoder.state_recon_loss(
                            ob_target,
                            state_target,
                            encoder_domain="target",
                            obs_domain="target",
                        )

                    cycle_loss = self._encoder.GAN_cycle_loss(
                        ob_target, ob_rec_target
                    ) + self._encoder.GAN_cycle_loss(ob_source, ob_rec_source)
                    G2_loss = self._encoder.GAN_generator_loss(
                        ob_gen_source, target_domain="source"
                    )

                    G_loss = self._encoder.GAN_generator_loss(ob_gen_target)
                    (
                        D_loss,
                        real_output,
                        gen_output,
                    ) = self._encoder.GAN_discriminator_loss(
                        ob_target, ob_gen_target, gradp=False
                    )

            losses["generator"].append(G_loss.detach().cpu().item())
            losses["cycle_consistency"].append(cycle_loss.detach().cpu().item())
            losses["generator2"].append(G2_loss.detach().cpu().item())
            if self._config.include_recon:
                if self._config.recon_state:
                    losses["GAN_state_reconstruction"].append(
                        recon_loss.detach().cpu().item()
                    )
                else:
                    losses["GAN_feature_reconstruction"].append(
                        feat_recon_loss.detach().cpu().item()
                    )
                    if not train:
                        losses["GAN_state_reconstruction"].append(
                            recon_loss.detach().cpu().item()
                        )

            losses["discriminator"].append(D_loss.detach().cpu().item())
            real_outputs.append(real_output.detach().cpu().item())
            gen_outputs.append(gen_output.detach().cpu().item())

        if train:
            self._encoder_G_lr_scheduler.step()
            self._encoder_D_lr_scheduler.step()

            info["encoder_discriminator_real_output"] = np.mean(real_outputs)
            info["encoder_discriminator_gen_output"] = np.mean(gen_outputs)
            print(
                "[TRAIN] Encoder Discriminator target Output: ",
                info["encoder_discriminator_real_output"],
            )
            print(
                "[TRAIN] Encoder Discriminator Gen Output: ",
                info["encoder_discriminator_gen_output"],
            )

            for k, v in losses.items():
                if len(v) > 0:
                    info["encoder_{}_loss".format(k)] = np.mean(v)
                    print(
                        "[TRAIN] Encoder {} Loss: ".format(k),
                        info["encoder_{}_loss".format(k)],
                    )
        else:
            for k, v in losses.items():
                if len(v) > 0:
                    info["eval_encoder_{}_loss".format(k)] = np.mean(v)
                    print(
                        "[TEST] Encoder {} Loss: ".format(k),
                        info["eval_encoder_{}_loss".format(k)],
                    )

        return info

    ### Auxiliary Functions ###

    def _define_optimizers(self):
        if self._config.include_recon:
            params = (
                list(self._encoder.shared_encoder.parameters())
                + list(self._encoder.source_encoder.parameters())
                + list(self._encoder.predict.parameters())
            )
            self._encoder_optim = optim.Adam(params, lr=3e-4)
            self._encoder_lr_scheduler = StepLR(
                self._encoder_optim, step_size=5, gamma=0.5,
            )
            optimizer_cuda(self._encoder_optim, self._config.device)

        epochs = self._config.max_stage_steps["supervised_pretraining"]
        lambda1 = lambda e: 0.1 if epochs == 0 else max((epochs - e) / epochs, 0.1)
        params = []
        for key, item in self._encoder.G.items():
            params += list(item.parameters())
        if self._config.include_recon:
            params += list(self._encoder.target_encoder.parameters())

        self._encoder_G_optim = optim.Adam(params, lr=0.0001, betas=(0.5, 0.999))
        self._encoder_G_lr_scheduler = LambdaLR(
            self._encoder_G_optim, lr_lambda=lambda1
        )
        optimizer_cuda(self._encoder_G_optim, self._config.device)
        params = []
        for key, item in self._encoder.D.items():
            params += list(item.parameters())

        self._encoder_D_optim = optim.Adam(params, lr=0.0004, betas=(0.5, 0.999))
        self._encoder_D_lr_scheduler = LambdaLR(
            self._encoder_D_optim, lr_lambda=lambda1
        )
        optimizer_cuda(self._encoder_D_optim, self._config.device)

    def _translate(self, ob, domain=None):
        ob = ob.copy()

        if domain == "target":
            return ob

        for k, v in ob.items():
            ob[k] = np.expand_dims(ob[k], axis=0)

        with torch.no_grad():
            ob = to_tensor(ob, self._config.device)
            gen_ob = self._encoder.generate(ob)
            gen_ob = gen_ob["ob"].detach().cpu().numpy().astype(np.uint8).squeeze()
        return {"ob": np.array(gen_ob)}

    def _generate_image_from_framestack(
        self,
        ob,
        env_type,
        reset,
        qpos=None,
        qvel=None,
        source_env=None,
        target_env=None,
        ob_stack=None,
    ):
        ob = ob.copy()
        if env_type == "source":
            ob_target = target_env.set_state_from_ob(qpos, qvel, reset=reset)
        if env_type == "target":
            ob_source = source_env.set_state_from_ob(qpos, qvel, reset=reset)
            if ob_stack is not None:
                ob = {"ob": np.concatenate((ob_stack, ob_source), axis=0)}
            else:
                ob = {"ob": np.concatenate((ob_source, ob_source, ob_source), axis=0)}

        obs_gen = self._translate(ob, "source")

        if env_type == "source":
            obs_gen = np.transpose(obs_gen["ob"][6:], (1, 2, 0))
            ob_real_img = np.transpose(ob_target, (1, 2, 0))
            return obs_gen.astype(np.uint8), ob_real_img.astype(np.uint8)

        if env_type == "target":
            obs_gen = np.transpose(obs_gen["ob"][6:], (1, 2, 0))
            ob_source_img = np.transpose(ob_source, (1, 2, 0))
            return (
                obs_gen.astype(np.uint8),
                ob_source_img.astype(np.uint8),
                ob["ob"][3:],
            )

    def _record_image(self, step, target_env):

        """
        Generates and logs examples of source env images and their translations.
        """

        if len(self._datasets["source"]) < 1:
            # datasets not yet populated
            return

        obs_save = []
        for i in range(10):
            index = np.random.randint(0, high=len(self._datasets["source"]))
            ob_source = self._datasets["source"][index]["ob"].copy()
            qposs = self._datasets["source"][index]["qpos"].copy()
            qvels = self._datasets["source"][index]["qvel"].copy()
            # state_source = self._datasets["source"][index]["state"]["state"].copy()

            obs_gen, obs_target = self._generate_image_from_framestack(
                ob_source,
                "source",
                reset=(i == 0),
                qpos=qposs,
                qvel=qvels,
                target_env=target_env,
            )
            obs_source = np.transpose(ob_source["ob"][6:], (1, 2, 0))
            obs_gen_diff = cv2.absdiff(obs_target, obs_gen)

            obs_target_random = self._datasets["target"][
                np.random.randint(0, high=len(self._datasets["target"]))
            ]["ob"].copy()
            obs_target_random = np.transpose(obs_target_random["ob"][6:], (1, 2, 0))
            obs_sample = np.concatenate(
                (obs_source, obs_gen, obs_target, obs_gen_diff, obs_target_random),
                axis=1,
            )
            obs_save.append(obs_sample)

        obs_save = np.concatenate(obs_save, axis=0)
        wandb.log({"GOT_image": [wandb.Image(obs_save)]}, step=step)

    def _record_video(
        self, obs, domain, qposs=None, qvels=None, source_env=None, target_env=None
    ):
        """
        Generates a video from frames in obs of source env obs, translated obs, target env obs, pixel-wise difference.
        """
        gen_frames = []
        if domain == "source":
            for (i, (framestack, qpos, qvel)) in enumerate(zip(obs[:-1], qposs, qvels)):
                obs_source = np.transpose(framestack["ob"][6:], (1, 2, 0))
                obs_gen, obs_target = self._generate_image_from_framestack(
                    framestack,
                    domain,
                    reset=(i == 0),
                    qpos=qpos,
                    qvel=qvel,
                    target_env=target_env,
                )
                obs_gen_diff = cv2.absdiff(obs_target, obs_gen)
                img = np.concatenate(
                    (obs_source, obs_gen, obs_target, obs_gen_diff), axis=1
                )
                gen_frames.append(img)
        elif domain == "target":
            ob_stack = None
            for (i, (framestack, qpos, qvel)) in enumerate(zip(obs[:-1], qposs, qvels)):
                obs_target = np.transpose(framestack["ob"][6:], (1, 2, 0))
                obs_gen, obs_source, ob_stack = self._generate_image_from_framestack(
                    framestack,
                    domain,
                    reset=(i == 0),
                    qpos=qpos,
                    qvel=qvel,
                    source_env=source_env,
                    ob_stack=ob_stack,
                )
                obs_gen_diff = cv2.absdiff(obs_target, obs_gen)
                img = np.concatenate(
                    (obs_source, obs_gen, obs_target, obs_gen_diff), axis=1
                )
                gen_frames.append(img)

        return gen_frames

    ### Misc Functions ###

    def _log_creation(self):
        if self._config.is_chef:
            logger.info("Creating a GOT model")
            logger.info(
                "The generator has %d parameters",
                count_parameters(self._encoder.G["target"]),
            )
            logger.info(
                "The discriminator has %d parameters",
                count_parameters(self._encoder.D["target"]),
            )
            if self._config.include_recon:
                logger.info(
                    "The encoder has %d parameters",
                    count_parameters(self._encoder.shared_encoder)
                    + count_parameters(self._encoder.target_encoder)
                    + count_parameters(self._encoder.source_encoder),
                )

    def sync_networks(self):
        sync_networks(self._encoder)

    def _network_cuda(self, device):
        self._encoder.to(device)

    def state_dict(self):
        return self._encoder.state_dict()

    def load_state_dict(self, ckpt):
        self._encoder.load_state_dict(ckpt)

    def set_encoder_requires_grad(self, req):
        self._encoder.set_encoder_requires_grad(req)
