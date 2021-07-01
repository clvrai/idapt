""" Define parameters for algorithms. """
import argparse

from .additional_configs import (
    add_additional_GOT_arguments,
    add_additional_garat_arguments,
)


def str2bool(v):
    return v.lower() == "true"


def str2intlist(value):
    if not value:
        return value
    else:
        value = value.strip("][")
        return [int(float(num.strip(""))) for num in value.split(",")]


def str2list(value):
    if not value:
        return value
    else:
        return [num for num in value.split(",")]


def create_parser():
    """
    Creates the argparser.  Use this to add additional arguments
    to the parser later.
    """
    parser = argparse.ArgumentParser(
        "DomainTransfer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        conflict_handler="resolve",
    )
    args, _ = parser.parse_known_args()

    # environment
    parser.add_argument("--source_env", type=str, default="SawyerPush-v0")
    parser.add_argument("--source_noise_bias", type=float, default=0.0)
    parser.add_argument("--source_noise_level", type=float, default=0.0)
    parser.add_argument("--source_ob_noise_level", type=float, default=0.0)

    parser.add_argument("--target_env", type=str, default="SawyerPush-v0")
    parser.add_argument("--target_noise_bias", type=float, default=0.0)
    parser.add_argument("--target_noise_level", type=float, default=0.0)
    parser.add_argument("--target_ob_noise_level", type=float, default=0.0)

    parser.add_argument("--envs", type=str2list, default=[])
    parser.add_argument("--eval_ckpt_paths", type=str2list, default=[])
    parser.add_argument("--early_term", type=str2bool, default=False)

    parser.add_argument("--seed", type=int, default=123)

    add_env_args(parser)

    add_method_arguments(parser)

    return parser


def add_method_arguments(parser):
    # algorithm
    parser.add_argument(
        "--algo",
        type=str,
        default="idapt",
        choices=["idapt", "asym_ac", "rasym_ac"],
    )

    # training
    parser.add_argument("--is_train", type=str2bool, default=True)
    args, _ = parser.parse_known_args()
    if not args.is_train:
        parser.add_argument(
            "--evaluator",
            type=str,
            default=None,
            choices=["multi_policy,", "multi_env"],
        )
    parser.add_argument("--resume", type=str2bool, default=True)
    parser.add_argument("--init_ckpt_path", type=str, default=None)
    parser.add_argument("--encoder_init_ckpt_path", type=str, default=None)
    parser.add_argument("--gpu", type=int, default=None)

    # evaluation
    parser.add_argument("--ckpt_num", type=int, default=None)
    parser.add_argument(
        "--num_eval", type=int, default=10, help="number of episodes for evaluation"
    )

    # environment
    try:
        parser.add_argument("--screen_width", type=int, default=100)
        parser.add_argument("--screen_height", type=int, default=100)
    except:
        pass
    parser.add_argument("--action_repeat", type=int, default=1)
    parser.add_argument("--target_freq", type=int, default=2)

    # misc
    parser.add_argument("--run_prefix", type=str, default=None)
    parser.add_argument("--notes", type=str, default="")

    # log
    parser.add_argument("--log_interval", type=int, default=1)
    parser.add_argument("--evaluate_interval", type=int, default=10)
    parser.add_argument("--ckpt_interval", type=int, default=200)
    parser.add_argument("--log_root_dir", type=str, default="log")
    parser.add_argument(
        "--wandb",
        type=str2bool,
        default=False,
        help="set it True if you want to use wandb",
    )
    parser.add_argument("--wandb_entity", type=str)
    parser.add_argument("--wandb_project", type=str)
    parser.add_argument("--record_video", type=str2bool, default=True)
    parser.add_argument("--record_video_caption", type=str2bool, default=True)
    try:
        parser.add_argument("--record_demo", type=str2bool, default=False)
    except:
        pass

    # observation normalization
    parser.add_argument("--ob_norm", type=str2bool, default=True)
    parser.add_argument("--max_ob_norm_step", type=int, default=int(1e6))
    parser.add_argument(
        "--clip_obs", type=float, default=200, help="the clip range of observation"
    )
    parser.add_argument(
        "--clip_range",
        type=float,
        default=5,
        help="the clip range after normalization of observation",
    )

    args, _ = parser.parse_known_args()

    parser.add_argument("--max_global_step", type=int, default=int(1e6))
    parser.add_argument("--batch_size", type=int, default=256)

    parser.add_argument("--recurrent", type=str2bool, default=False)

    add_policy_arguments(parser)
    if args.algo == "idapt":
        add_idapt_arguments(parser)
        print("idapt args:", parser.parse_args())
    elif args.algo == "rasym_ac" or args.algo == "asym_ac":
        add_asym_ac_arguments(parser)


def add_policy_arguments(parser):
    # network
    parser.add_argument("--log_std_min", type=float, default=-4)
    parser.add_argument("--log_std_max", type=float, default=2)
    parser.add_argument("--policy_mlp_dim", type=str2intlist, default=[256, 256])
    parser.add_argument("--critic_mlp_dim", type=str2intlist, default=[256, 256])
    parser.add_argument("--critic_ensemble", type=int, default=1)
    parser.add_argument(
        "--policy_activation", type=str, default="relu", choices=["relu", "elu", "tanh"]
    )
    parser.add_argument("--tanh_policy", type=str2bool, default=True)
    parser.add_argument("--gaussian_policy", type=str2bool, default=True)
    parser.add_argument("--learn_std", type=str2bool, default=True)
    # encoder
    parser.add_argument(
        "--encoder_type", type=str, default="cnn", choices=["mlp", "cnn"]
    )
    parser.add_argument("--encoder_mlp_dim", type=str2intlist, default=[256, 256])
    parser.add_argument("--encoder_image_size", type=int, default=92)
    parser.add_argument("--encoder_conv_dim", type=int, default=32)
    parser.add_argument("--encoder_kernel_size", type=str2intlist, default=[3, 3, 3, 3])
    parser.add_argument("--encoder_stride", type=str2intlist, default=[2, 1, 1, 1])
    parser.add_argument("--encoder_conv_output_dim", type=int, default=50)
    parser.add_argument("--encoder_soft_update_weight", type=float, default=0.95)
    parser.add_argument("--encoder_domain_layers", type=int, default=2)
    parser.add_argument("--greyscale", type=str2bool, default=False)
    parser.add_argument("--colorjitter", type=str2bool, default=False)
    parser.add_argument("--colorjitter_narrow", type=str2bool, default=False)
    args, _ = parser.parse_known_args()
    if args.encoder_type == "cnn":
        parser.set_defaults(screen_width=100, screen_height=100)
        parser.set_defaults(policy_mlp_dim=[1024, 1024])
        parser.set_defaults(critic_mlp_dim=[1024, 1024])
    parser.add_argument("--run_rl_from_state", type=str2bool, default=False)

    parser.add_argument("--discriminator_mlp_dim", type=str2intlist, default=[256, 256])
    parser.add_argument("--discriminator_activation", type=str, default="relu")
    parser.add_argument(
        "--discriminator_lr",
        type=float,
        default=3e-4,
        help="the learning rate of the discriminator",
    )

    # actor-critic
    parser.add_argument(
        "--actor_lr", type=float, default=3e-4, help="the learning rate of the actor"
    )
    parser.add_argument(
        "--critic_lr", type=float, default=3e-4, help="the learning rate of the critic"
    )
    parser.add_argument(
        "--critic_soft_update_weight",
        type=float,
        default=0.995,
        help="the average coefficient",
    )
    parser.add_argument(
        "--encoder_lr",
        type=float,
        default=3e-4,
        help="the learning rate of the encoder",
    )

    parser.add_argument("--absorbing_state", type=str2bool, default=False)

    add_multistage_arguments(parser)

    return parser


def add_rl_arguments(parser):
    parser.add_argument(
        "--rl_discount_factor", type=float, default=0.99, help="the discount factor"
    )
    parser.add_argument("--warm_up_steps", type=int, default=0)


def add_on_policy_arguments(parser):
    parser.add_argument("--rollout_length", type=int, default=2000)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--advantage_norm", type=str2bool, default=True)


def add_off_policy_arguments(parser):
    parser.add_argument(
        "--buffer_size", type=int, default=int(1e6), help="the size of the buffer"
    )
    parser.set_defaults(warm_up_steps=1000)


def add_sac_arguments(parser):
    add_rl_arguments(parser)
    add_off_policy_arguments(parser)

    parser.add_argument("--reward_scale", type=float, default=1.0, help="reward scale")
    parser.add_argument("--actor_update_freq", type=int, default=2)
    parser.add_argument("--critic_target_update_freq", type=int, default=2)
    parser.add_argument("--target_entropy", type=float, default=None)
    parser.add_argument("--max_grad_norm", type=float, default=None)
    parser.add_argument("--alpha_init_temperature", type=float, default=0.1)
    parser.add_argument(
        "--alpha_lr", type=float, default=1e-4, help="the learning rate of the actor"
    )
    parser.set_defaults(actor_lr=3e-4)
    parser.set_defaults(critic_lr=3e-4)
    parser.set_defaults(evaluate_interval=5000)
    parser.set_defaults(ckpt_interval=10000)
    parser.set_defaults(log_interval=500)
    parser.set_defaults(critic_soft_update_weight=0.99)
    parser.set_defaults(buffer_size=100000)
    parser.set_defaults(critic_ensemble=2)
    parser.set_defaults(ob_norm=False)


def add_ppo_arguments(parser):
    add_rl_arguments(parser)
    add_on_policy_arguments(parser)

    parser.add_argument("--ppo_clip", type=float, default=0.2)
    parser.add_argument("--value_loss_coeff", type=float, default=0.5)
    parser.add_argument("--action_loss_coeff", type=float, default=1.0)
    parser.add_argument("--entropy_loss_coeff", type=float, default=1e-4)

    parser.add_argument("--ppo_epoch", type=int, default=5)
    parser.add_argument("--max_grad_norm", type=float, default=None)
    parser.add_argument("--actor_update_freq", type=int, default=1)
    parser.set_defaults(ob_norm=True)
    parser.set_defaults(evaluate_interval=20)
    parser.set_defaults(ckpt_interval=20)


def add_il_arguments(parser):
    parser.add_argument("--demo_path", type=str, default=None, help="path to demos")
    parser.add_argument(
        "--demo_low_level",
        type=str2bool,
        default=False,
        help="use low level actions for training",
    )
    parser.add_argument(
        "--demo_subsample_interval",
        type=int,
        default=1,
        # default=20, # used in GAIL
        help="subsample interval of expert transitions",
    )
    parser.add_argument(
        "--demo_sample_range_start", type=float, default=0.0, help="sample demo range"
    )
    parser.add_argument(
        "--demo_sample_range_end", type=float, default=1.0, help="sample demo range"
    )


def add_gail_arguments(parser):
    parser.add_argument("--gail_entropy_loss_coeff", type=float, default=0.0)
    parser.add_argument(
        "--gail_reward", type=str, default="vanilla", choices=["vanilla", "gan", "d"]
    )
    parser.add_argument("--discriminator_lr", type=float, default=1e-4)
    parser.add_argument("--discriminator_mlp_dim", type=str2intlist, default=[256, 256])
    parser.add_argument(
        "--discriminator_activation",
        type=str,
        default="tanh",
        choices=["relu", "elu", "tanh"],
    )
    parser.add_argument("--discriminator_update_freq", type=int, default=4)
    parser.add_argument("--gail_no_action", type=str2bool, default=False)
    parser.add_argument("--gail_env_reward", type=float, default=0.0)
    parser.add_argument("--gail_grad_penalty_coeff", type=float, default=10.0)

    parser.add_argument(
        "--gail_rl_algo", type=str, default="ppo", choices=["ppo", "sac", "td3"]
    )


def add_gaifo_arguments(parser):
    parser.add_argument("--gaifo_discriminator_lr", type=float, default=1e-2)
    parser.add_argument("--gaifo_discriminator_weight_decay", type=float, default=1e-3)
    parser.add_argument(
        "--gaifo_discriminator_mlp_dim", type=str2intlist, default=[256, 256]
    )
    parser.add_argument("--gaifo_discriminator_activation", type=str, default="relu")
    parser.add_argument(
        "--gaifo_reward",
        type=str,
        default="nonsatgan",
        choices=["vanilla", "gan", "d", "nonsatgan"],
    )
    parser.add_argument("--gaifo_discriminator_batch_size", type=int, default=256)
    parser.add_argument("--gaifo_discriminator_update_freq", type=int, default=1)
    parser.add_argument("--gaifo_agent_update_freq", type=int, default=1)
    parser.add_argument("--gaifo_entropy_loss_coeff", type=float, default=0.1)
    parser.add_argument("--gaifo_grad_penalty_coeff", type=float, default=10.0)
    parser.add_argument("--gaifo_env_reward", type=float, default=0.0)
    parser.add_argument("--gaifo_rl_discount_factor", type=float, default=0.99)
    parser.add_argument("--ifo_actor_lr", type=float, default=3e-5)
    parser.add_argument("--ifo_critic_lr", type=float, default=3e-5)


def add_multistage_arguments(parser):
    parser.set_defaults(trainer_type="MST")

    parser.add_argument("--ckpt_stage_only", type=str2bool, default=True)
    parser.add_argument("--target_transitions_count", type=int, default=1000)

    parser.add_argument(
        "--max_stage_steps",
        type=eval,
        default={
            "supervised_pretraining": 80,
            "policy_init": 5e5,
            "grounding": 50000,
            "supervised_training": 10,
            "policy_training": 2e5,
        },
    )

    ### Per Stage Logging/Evaluation Args ###

    parser.add_argument("--policy_init_evaluate_interval", type=int, default=5000)
    parser.add_argument("--policy_init_ckpt_interval", type=int, default=20000)
    parser.add_argument("--policy_init_log_interval", type=int, default=500)
    parser.add_argument("--policy_init_warm_up_steps", type=int, default=1000)

    parser.add_argument("--policy_training_evaluate_interval", type=int, default=5000)
    parser.add_argument("--policy_training_ckpt_interval", type=int, default=100000)
    parser.add_argument("--policy_training_log_interval", type=int, default=500)
    parser.add_argument("--policy_training_warm_up_steps", type=int, default=0)

    parser.add_argument("--grounding_evaluate_interval", type=int, default=5)
    parser.add_argument("--grounding_ckpt_interval", type=int, default=20)
    parser.add_argument("--grounding_log_interval", type=int, default=1)
    parser.add_argument("--grounding_warm_up_steps", type=int, default=0)

    parser.add_argument("--supervised_evaluate_interval", type=int, default=5)
    parser.add_argument("--supervised_ckpt_interval", type=int, default=20)
    parser.add_argument("--supervised_log_interval", type=int, default=1)
    parser.add_argument("--supervised_warm_up_steps", type=int, default=0)


def add_env_args(parser):
    """
    Adds a list of arguments to argparser for the sawyer and fetchreach environment.
    """
    # sawyer
    parser.add_argument(
        "--reward_type",
        type=str,
        default="dense",
        choices=["dense", "sparse"],
        help="reward type",
    )
    parser.add_argument(
        "--distance_threshold",
        type=float,
        default=0.06,
        help="distance threshold for termination",
    )
    parser.add_argument(
        "--max_episode_steps",
        type=int,
        default=70,
        help="maximum timesteps in an episode",
    )
    parser.add_argument(
        "--camera_name",
        type=str,
        default="visview",
        help="camera name in an environment",
    )

    # observations
    parser.add_argument(
        "--frame_skip", type=int, default=1, help="Numer of skip frames"
    )
    parser.add_argument(
        "--action_repeat", type=int, default=1, help="number of action repeats"
    )
    parser.add_argument(
        "--ctrl_reward_coef", type=float, default=0, help="control reward coefficient"
    )

    parser.add_argument(
        "--kp", type=float, default=40.0, help="p term for a PID controller"
    )  # 150.)
    parser.add_argument(
        "--kd", type=float, default=8.0, help="d term for a PID controller"
    )  # 20.)
    parser.add_argument(
        "--ki", type=float, default=0.0, help="i term for a PID controller"
    )
    parser.add_argument(
        "--frame_dt", type=float, default=0.15, help="delta t between each frame"
    )  # 0.1)
    parser.add_argument(
        "--use_robot_indicator",
        type=eval,
        default=False,
        help="enable visualization of robot indicator for motion planner",
    )
    parser.add_argument(
        "--use_target_robot_indicator",
        type=eval,
        default=False,
        help="enable visualization of robot indicator for target position of motion planner",
    )
    parser.add_argument(
        "--success_reward", type=float, default=150.0, help="completion reward"
    )
    parser.add_argument(
        "--contact_threshold",
        type=float,
        default=-0.002,
        help="depth thredhold for contact",
    )
    parser.add_argument(
        "--joint_margin", type=float, default=0.001, help="marin of each joint"
    )
    parser.add_argument("--task_level", type=str, default="easy")
    parser.add_argument(
        "--step_size",
        type=float,
        default=0.02,
        help="step size for invalid target handling",
    )
    # puck
    parser.add_argument("--puck_friction", type=float, default=2.0)
    parser.add_argument("--puck_mass", type=float, default=0.01)
    parser.add_argument("--source_env_puck_friction", type=float, default=2.0)
    parser.add_argument("--source_env_puck_mass", type=float, default=0.01)
    parser.add_argument("--target_env_puck_friction", type=float, default=2.0)
    parser.add_argument("--target_env_puck_mass", type=float, default=0.01)

    parser.add_argument("--env_ob_source", type=str2bool, default=False)
    parser.add_argument("--end_effector", type=str2bool, default=True)
    parser.add_argument("--ik_target", type=str, default="grip_site")
    parser.add_argument(
        "--action_range", type=float, default=0.1, help="range of radian"
    )
    parser.add_argument("--dr", type=str2bool, default=False)
    parser.add_argument("--dr_params_set", type=str, default="IP_large_range")

    parser.add_argument("--mod_env_params", type=str2bool, default=False)
    parser.add_argument("--param_mod_instructions", type=eval, default=[])

    parser.add_argument("--unity", type=str2bool, default=False)
    parser.add_argument("--unity_editor", type=str2bool, default=False)
    parser.add_argument("--virtual_display", type=str, default=":1")
    parser.add_argument("--port", type=int, default=4000)

    # FetchReach action
    parser.add_argument("--action_rotation_degrees", type=float, default=0.0)
    parser.add_argument("--action_z_bias", type=float, default=0.0)


def add_GOT_arguments(parser):
    parser.add_argument("--GOT_batch_size", type=int, default=8)
    parser.add_argument("--source_encoder_train_epoch", type=int, default=40)
    parser.add_argument("--source_encoder_finetune_epoch", type=int, default=5)
    parser.add_argument("--recon_coeff", type=float, default=5.0)
    parser.add_argument("--recon_state", type=str2bool, default=True)
    parser.add_argument("--include_recon", type=str2bool, default=True)
    parser.add_argument("--cycle_coeff", type=float, default=1.0)
    parser.add_argument("--gather_random_rollouts", type=str2bool, default=True)
    parser.add_argument("--accumulate_data", type=str2bool, default=False)
    parser.add_argument("--save_data", type=str2bool, default=False)
    parser.add_argument(
        "--target_demo_path", type=str, default=None, help="path to target env demos"
    )
    parser.add_argument(
        "--source_demo_path", type=str, default=None, help="path to source env demos"
    )
    parser.add_argument(
        "--target_eval_demo_path",
        type=str,
        default=None,
        help="path to target env demos",
    )
    parser.add_argument(
        "--source_eval_demo_path",
        type=str,
        default=None,
        help="path to source env demos",
    )
    parser.add_argument("--num_transitions", type=int, default=1000)
    parser.add_argument("--num_eval_transitions", type=int, default=100)
    parser.add_argument(
        "--demo_low_level",
        type=str2bool,
        default=False,
        help="use low level actions for training",
    )
    parser.add_argument(
        "--demo_subsample_interval",
        type=int,
        default=1,
        # default=20, # used in GAIL
        help="subsample interval of expert transitions",
    )
    parser.add_argument("--data", type=str, default="random")

    add_additional_GOT_arguments(parser)


def add_idapt_arguments(parser):
    add_ppo_arguments(parser)
    add_gaifo_arguments(parser)
    add_il_arguments(parser)

    ### Default Multi-stage Training Args ###

    parser.set_defaults(warm_up_steps=0)
    parser.set_defaults(max_global_step=1e6)
    parser.add_argument("--grounding_steps", type=int, default=5)
    parser.add_argument(
        "--mode",
        type=str,
        default="cross_visual_physics",
        choices=["cross_physics", "cross_visual", "cross_visual_physics"],
    )
    parser.add_argument("--noisy_baseline", type=str2bool, default=False)

    ### AT Training Args ###

    parser.add_argument("--smoothing_factor", type=float, default=0.95)
    parser.add_argument("--load_AT", type=str2bool, default=False)
    parser.add_argument("--AT_acdiff", type=str2bool, default=True)

    args, unparsed = parser.parse_known_args()

    if "visual" in args.mode:
        add_GOT_arguments(parser)

    add_sac_arguments(parser)

    add_additional_garat_arguments(parser)


def add_asym_ac_arguments(parser):
    add_sac_arguments(parser)

    parser.set_defaults(recurrent=True)
    parser.add_argument("--rnn_dim", type=int, default=128)
    parser.add_argument("--rnn_seq_length", type=int, default=2)
    parser.add_argument("--burn_in", type=int, default=0)

    args, unparsed = parser.parse_known_args()

    parser.set_defaults(policy_init_evaluate_interval=args.evaluate_interval)
    parser.set_defaults(policy_init_ckpt_interval=args.ckpt_interval)
    parser.set_defaults(policy_init_log_interval=args.log_interval)
    parser.set_defaults(policy_init_warm_up_steps=args.warm_up_steps)


def add_id_arguments(parser):
    add_rl_arguments(parser)
    add_on_policy_arguments(parser)

    parser.set_defaults(warm_up_steps=1000)
    parser.add_argument(
        "--action_pred_mlp_dim", type=str2intlist, default=[256, 256, 256, 256]
    )
    parser.add_argument(
        "--predictor_lr",
        type=float,
        default=3e-4,
        help="the learning rate of the discriminator",
    )
    parser.add_argument("--confusion_coeff", type=float, default=1e-2)
    parser.add_argument("--dis_train_period_multiplyer", type=float, default=1.0)
    parser.add_argument("--predictor_train_period", type=int, default=16)


def argparser():
    """
    Directly parses the arguments
    """
    parser = create_parser()
    args, unparsed = parser.parse_known_args()

    return args, unparsed
