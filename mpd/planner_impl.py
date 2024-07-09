import torch
import einops
import mlflow
from math import ceil

from corallab_lib.task import Task

from corallab_planners.backends.planner_interface import PlannerInterface

from torch_robotics.torch_utils.torch_utils import DEFAULT_TENSOR_ARGS, freeze_torch_model_params

from mp_baselines.planners.costs.cost_functions import (
    CostCollision,
    CostComposite,
    CostGPTrajectory,
    CostSmoothnessCHOMP
)

from mpd.models.diffusion_models.guides import GuideManagerTrajectoriesWithVelocity
from mpd.models.diffusion_models.sample_functions import guide_gradient_steps
from mpd.trainer import get_dataset


class MPDPlanner(PlannerInterface):
    def __init__(
            self,
            planner_name : str,
            task : Task = None,
            planner_alg: str = 'diffusion_prior',

            start_guide_steps_fraction: float = 0.25,
            n_guide_steps: int = 5,
            n_diffusion_steps_without_noise: int = 5,

            trajectory_duration: float = 5.0,  # currently fixed

            model_training_run_id : str = None,

            tensor_args : dict = DEFAULT_TENSOR_ARGS,
            **kwargs
    ):
        # Vars
        self.tensor_args = tensor_args
        self.run_prior_only = False
        self.run_prior_then_guidance = False
        if planner_alg == 'mpd':
            pass
        elif planner_alg == 'diffusion_prior_then_guide':
            self.run_prior_then_guidance = True
        elif planner_alg == 'diffusion_prior':
            self.run_prior_only = True
        else:
            raise NotImplementedError

        self.ctask = task

        # TODO: Adapt to use general task wrapper
        self.dataset = self._get_dataset(model_training_run_id)
        self.n_support_points = self.dataset.n_support_points
        self.env = self.dataset.env
        self.robot = self.dataset.robot
        self.task = self.dataset.task

        # Check that the model was trained on this problem
        assert self.ctask.env.name == self.dataset.env.name
        assert self.ctask.robot.name == self.dataset.robot.name

        self.dt = trajectory_duration / self.n_support_points  # time interval for finite differences
        self.robot.dt = self.dt
        self.n_diffusion_steps_without_noise = n_diffusion_steps_without_noise

        self.model = self._get_model(model_training_run_id)

        # Trajectory Guidance
        self.guide = self._create_traj_guide()

        self.t_start_guide = ceil(start_guide_steps_fraction * self.model.n_diffusion_steps)
        self.n_guide_steps = n_guide_steps
        self.sample_fn_kwargs = dict(
            guide=None if self.run_prior_then_guidance or self.run_prior_only else guide,
            n_guide_steps=self.n_guide_steps,
            t_start_guide=self.t_start_guide,
            noise_std_extra_schedule_fn=lambda x: 0.5,
            task=task,
        )

    @property
    def name(self):
        return "mpd_planner"

    def _get_dataset(self, training_run_id):
        "Need the dataset for normalizing inputs for and outputs from model"

        training_run = mlflow.get_run(training_run_id)
        training_params = training_run.data.params

        # Load dataset with env, robot, task
        train_subset, train_dataloader, val_subset, val_dataloader = get_dataset(
            dataset_subdir=training_params["dataset_subdir"],
            dataset_class=training_params["dataset_class"],
            include_velocity=training_params["include_velocity"],
            normalizer='SafeLimitsNormalizer',
            # other
            use_extra_objects=False,
            obstacle_cutoff_margin=0.05,
            # **args,
            tensor_args=self.tensor_args
        )

        return train_subset.dataset

    def _get_model(self, training_run_id):
        logged_model_path = f'runs:/{training_run_id}/model'
        model = mlflow.pytorch.load_model(logged_model_path)
        freeze_torch_model_params(model)
        model = torch.compile(model)
        return model

    def _create_traj_guide(
            self,
            use_guide_on_extra_objects_only : bool = False,
            weight_grad_cost_collision : float = 1e-2,
            weight_grad_cost_smoothness: float = 1e-7,
            factor_num_interpolated_points_for_collision: float = 1.5,
    ):
        # Cost collisions
        cost_collision_l = []
        weights_grad_cost_l = []  # for guidance, the weights_cost_l are the gradient multipliers (after gradient clipping)
        if use_guide_on_extra_objects_only:
            collision_fields = self.task.get_collision_fields_extra_objects()
        else:
            collision_fields = self.task.get_collision_fields()

        for collision_field in collision_fields:
            cost_collision_l.append(
                CostCollision(
                    self.robot, self.n_support_points,
                    field=collision_field,
                    sigma_coll=1.0,
                    tensor_args=self.tensor_args
                )
            )
            weights_grad_cost_l.append(weight_grad_cost_collision)

        # Cost smoothness
        cost_smoothness_l = [
            CostGPTrajectory(
                self.robot, self.n_support_points, self.dt, sigma_gp=1.0,
                tensor_args=self.tensor_args
            )
        ]
        weights_grad_cost_l.append(weight_grad_cost_smoothness)

        ####### Cost composition
        cost_func_list = [
            *cost_collision_l,
            *cost_smoothness_l,
        ]

        cost_composite = CostComposite(
            self.robot, self.n_support_points, cost_func_list,
            weights_cost_l=weights_grad_cost_l,
            tensor_args=self.tensor_args
        )

        guide = GuideManagerTrajectoriesWithVelocity(
            self.dataset,
            cost_composite,
            clip_grad=True,
            interpolate_trajectories_for_collision=True,
            num_interpolated_points=ceil(self.n_support_points * factor_num_interpolated_points_for_collision),
            tensor_args=self.tensor_args,
        )

        return guide

    def solve(
            self,
            start_state_pos,
            goal_state_pos,
            n_samples=1,
            **kwargs
    ):
        hard_conds = self.dataset.get_hard_conditions(
            torch.vstack((start_state_pos, goal_state_pos)), normalize=True
        )
        start_state = torch.hstack((start_state_pos, torch.zeros_like(start_state_pos))).unsqueeze(0)
        context = None

        trajs_normalized_iters = self.model.run_inference(
            context, hard_conds,
            n_samples=n_samples, horizon=self.n_support_points,
            return_chain=True,
            **self.sample_fn_kwargs,
            n_diffusion_steps_without_noise=self.n_diffusion_steps_without_noise,
            # ddim=True
        )

        if self.run_prior_then_guidance:
            n_post_diffusion_guide_steps = (self.t_start_guide + self.n_diffusion_steps_without_noise) * self.n_guide_steps
            trajs = trajs_normalized_iters[-1]

            trajs_post_diff_l = []

            for i in range(n_post_diffusion_guide_steps):
                trajs = guide_gradient_steps(
                    trajs,
                    hard_conds=hard_conds,
                    guide=self.guide,
                    n_guide_steps=1,
                    unnormalize_data=False,
                )

                trajs_post_diff_l.append(trajs)

            chain = torch.stack(trajs_post_diff_l, dim=1)
            chain = einops.rearrange(chain, 'b post_diff_guide_steps h d -> post_diff_guide_steps b h d')
            trajs_normalized_iters = torch.cat((trajs_normalized_iters, chain))

        trajs_iters = self.dataset.unnormalize_trajectories(trajs_normalized_iters)
        solution = trajs_iters[-1]
        info = { "solution_iters": trajs_iters }

        return solution, info

    def reset(self):
        pass
