def add_additional_GOT_arguments(parser):
    args, unparsed = parser.parse_known_args()

    ### Source Envs ###
    if args.source_env == "InvertedPendulum-v2" and args.data == "random":
        parser.set_defaults(
            source_demo_path="data/InvertedPendulum-v2_image_random_20001_transitions.pkl"
        )
        parser.set_defaults(
            source_eval_demo_path="data/InvertedPendulum-v2_image_random_2001_transitions.pkl"
        )
    elif args.source_env == "HalfCheetah-v3":
        if args.data == "backwards":
            parser.set_defaults(
                source_demo_path="data/HalfCheetah-v3_image_backwards_20001_transitions.pkl"
            )
            parser.set_defaults(
                source_eval_demo_path="data/HalfCheetah-v3_image_backwards_2001_transitions.pkl"
            )
        elif args.data == "random":
            parser.set_defaults(
                source_demo_path="data/HalfCheetah-v3_image_random_20001_transitions.pkl"
            )
            parser.set_defaults(
                source_eval_demo_path="data/HalfCheetah-v3_image_random_2001_transitions.pkl"
            )
        elif args.data == "expert":
            parser.set_defaults(
                source_demo_path="data/HalfCheetah-v3_image_expert_20001_transitions.pkl"
            )
            parser.set_defaults(
                source_eval_demo_path="data/HalfCheetah-v3_image_expert_2001_transitions.pkl"
            )
        elif args.data == "random_expert":
            parser.set_defaults(
                source_demo_path="data/HalfCheetah-v3_image_random_expert_20001_transitions.pkl"
            )
            parser.set_defaults(
                source_eval_demo_path="data/HalfCheetah-v3_image_expert_2001_transitions.pkl"
            )
        elif args.data == "mixed":
            parser.set_defaults(
                source_demo_path="data/HalfCheetah-v3_image_mixed_20001_transitions.pkl"
            )
            parser.set_defaults(
                source_eval_demo_path="data/HalfCheetah-v3_image_expert_2001_transitions.pkl"
            )
    elif args.source_env == "GymWalker-v0":
        if args.data == "backwards":
            parser.set_defaults(
                source_demo_path="data/GymWalker-v0_image_backwards_20001_transitions.pkl"
            )
            parser.set_defaults(
                source_eval_demo_path="data/GymWalker-v0_image_backwards_2001_transitions.pkl"
            )
        elif args.data == "random":
            parser.set_defaults(
                source_demo_path="data/GymWalker-v0_image_random_20001_transitions.pkl"
            )
            parser.set_defaults(
                source_eval_demo_path="data/GymWalker-v0_image_random_2001_transitions.pkl"
            )
        elif args.data == "expert":
            parser.set_defaults(
                source_demo_path="data/GymWalker-v0_image_expert_20001_transitions.pkl"
            )
            parser.set_defaults(
                source_eval_demo_path="data/GymWalker-v0_image_expert_2001_transitions.pkl"
            )
        elif args.data == "random_expert":
            parser.set_defaults(
                source_demo_path="data/GymWalker-v0_image_random_expert_20001_transitions.pkl"
            )
            parser.set_defaults(
                source_eval_demo_path="data/GymWalker-v0_image_expert_2001_transitions.pkl"
            )
        elif args.data == "mixed":
            parser.set_defaults(
                source_demo_path="data/GymWalker-v0_image_mixed_20001_transitions.pkl"
            )
            parser.set_defaults(
                source_eval_demo_path="data/GymWalker-v0_image_expert_2001_transitions.pkl"
            )
    elif args.source_env == "FetchReach-v1" and args.data == "random":
        parser.set_defaults(
            source_demo_path="data/FetchReach-v1_image_random_20001_transitions.pkl"
        )
        parser.set_defaults(
            source_eval_demo_path="data/FetchReach-v1_image_random_2001_transitions.pkl"
        )
    elif args.source_env == "SawyerPushZoom-v0" and args.data == "random":
        parser.set_defaults(
            source_demo_path="data/SawyerPushZoom-v0_image_random_20001_transitions.pkl"
        )
        parser.set_defaults(
            source_eval_demo_path="data/SawyerPushZoom-v0_image_random_2001_transitions.pkl"
        )

    ### Target Envs ###
    if "GymInvertedPendulumEasy" in args.target_env and args.data == "random":
        parser.set_defaults(
            target_demo_path="data/GymInvertedPendulumEasy-v0_image_random_20001_transitions.pkl"
        )
        parser.set_defaults(
            target_eval_demo_path="data/GymInvertedPendulumEasy-v0_image_random_2001_transitions.pkl"
        )
    elif "GymInvertedPendulumDM" in args.target_env and args.data == "random":
        parser.set_defaults(
            target_demo_path="data/GymInvertedPendulumDM-v0_image_random_20001_transitions.pkl"
        )
        parser.set_defaults(
            target_eval_demo_path="data/GymInvertedPendulumDM-v0_image_random_2001_transitions.pkl"
        )
    elif "GymHalfCheetahEasy" in args.target_env and args.data == "backwards":
        parser.set_defaults(
            target_demo_path="data/GymHalfCheetahEasy-v0_image_backwards_20001_transitions.pkl"
        )
        parser.set_defaults(
            target_eval_demo_path="data/GymHalfCheetahEasy-v0_image_backwards_2001_transitions.pkl"
        )
    elif "GymHalfCheetahDM" in args.target_env:
        if args.data == "backwards":
            parser.set_defaults(
                target_demo_path="data/GymHalfCheetahDM-v0_image_backwards_20001_transitions.pkl"
            )
            parser.set_defaults(
                target_eval_demo_path="data/GymHalfCheetahDM-v0_image_backwards_2001_transitions.pkl"
            )
        elif args.data == "random":
            parser.set_defaults(
                target_demo_path="data/GymHalfCheetahDM-v0_image_random_20001_transitions.pkl"
            )
            parser.set_defaults(
                target_eval_demo_path="data/GymHalfCheetahDM-v0_image_random_2001_transitions.pkl"
            )
        elif args.data == "expert":
            parser.set_defaults(
                target_demo_path="data/GymHalfCheetahDM-v0_image_expert_20001_transitions.pkl"
            )
            parser.set_defaults(
                target_eval_demo_path="data/GymHalfCheetahDM-v0_image_expert_2001_transitions.pkl"
            )
        elif args.data == "random_expert":
            parser.set_defaults(
                target_demo_path="data/GymHalfCheetahDM-v0_image_random_expert_20001_transitions.pkl"
            )
            parser.set_defaults(
                target_eval_demo_path="data/GymHalfCheetahDM-v0_image_expert_2001_transitions.pkl"
            )
        elif args.data == "mixed":
            parser.set_defaults(
                target_demo_path="data/GymHalfCheetahDM-v0_image_mixed_20001_transitions.pkl"
            )
            parser.set_defaults(
                target_eval_demo_path="data/GymHalfCheetahDM-v0_image_expert_2001_transitions.pkl"
            )
    elif "GymWalkerEasy" in args.target_env and args.data == "backwards":
        parser.set_defaults(
            target_demo_path="data/GymWalkerEasy-v0_image_backwards_20001_transitions.pkl"
        )
        parser.set_defaults(
            target_eval_demo_path="data/GymWalkerEasy-v0_image_backwards_2001_transitions.pkl"
        )
    elif "GymWalkerDM" in args.target_env:
        if args.data == "backwards":
            parser.set_defaults(
                target_demo_path="data/GymWalkerDM-v0_image_backwards_20001_transitions.pkl"
            )
            parser.set_defaults(
                target_eval_demo_path="data/GymWalkerDM-v0_image_backwards_2001_transitions.pkl"
            )
        elif args.data == "random":
            parser.set_defaults(
                target_demo_path="data/GymWalkerDM-v0_image_random_20001_transitions.pkl"
            )
            parser.set_defaults(
                target_eval_demo_path="data/GymWalkerDM-v0_image_random_2001_transitions.pkl"
            )
        elif args.data == "expert":
            parser.set_defaults(
                target_demo_path="data/GymWalkerDM-v0_image_expert_20001_transitions.pkl"
            )
            parser.set_defaults(
                target_eval_demo_path="data/GymWalkerDM-v0_image_expert_2001_transitions.pkl"
            )
        elif args.data == "random_expert":
            parser.set_defaults(
                target_demo_path="data/GymWalkerDM-v0_image_random_expert_20001_transitions.pkl"
            )
            parser.set_defaults(
                target_eval_demo_path="data/GymWalkerDM-v0_image_expert_2001_transitions.pkl"
            )
        elif args.data == "mixed":
            parser.set_defaults(
                target_demo_path="data/GymWalkerDM-v0_image_mixed_20001_transitions.pkl"
            )
            parser.set_defaults(
                target_eval_demo_path="data/GymWalkerDM-v0_image_expert_2001_transitions.pkl"
            )
    elif args.target_env == "GymFetchReach-v0" and args.data == "random":
        parser.set_defaults(
            target_demo_path="data/GymFetchReach-v0_image_random_20001_transitions.pkl"
        )
        parser.set_defaults(
            target_eval_demo_path="data/GymFetchReach-v0_image_random_2001_transitions.pkl"
        )
    elif args.target_env == "GymFetchReach2-v0" and args.data == "random":
        parser.set_defaults(
            target_demo_path="data/GymFetchReach2-v0_image_random_20001_transitions.pkl"
        )
        parser.set_defaults(
            target_eval_demo_path="data/GymFetchReach2-v0_image_random_2001_transitions.pkl"
        )
    elif args.target_env == "SawyerPushZoomEasy-v0" and args.data == "random":
        parser.set_defaults(
            target_demo_path="data/SawyerPushZoomEasy-v0_image_random_20001_transitions.pkl"
        )
        parser.set_defaults(
            target_eval_demo_path="data/SawyerPushZoomEasy-v0_image_random_2001_transitions.pkl"
        )
    elif (
        args.target_env == "SawyerPushShiftViewZoomBackground-v0"
        and args.data == "random"
    ):
        parser.set_defaults(
            target_demo_path="data/SawyerPushShiftViewZoomBackground-v0_image_random_20001_transitions.pkl"
        )
        parser.set_defaults(
            target_eval_demo_path="data/SawyerPushShiftViewZoomBackground-v0_image_random_2001_transitions.pkl"
        )


def add_additional_garat_arguments(parser):
    args, unparsed = parser.parse_known_args()

    if (
        "Hopper" in args.source_env
        or "Walker" in args.source_env
        or "HalfCheetah" in args.source_env
    ):
        ### Hopper/Walker ###
        parser.set_defaults(action_repeat=2)
        parser.set_defaults(frame_skip=2)
        parser.set_defaults(early_term=True)
        parser.set_defaults(
            max_stage_steps={
                "supervised_pretraining": 0,
                "policy_init": 0,
                "grounding": 50000,
                "supervised_training": 10,
                "policy_training": 1e5,
            }
        )
        if args.run_prefix == "more_grounding":
            parser.set_defaults(
                max_stage_steps={
                    "supervised_pretraining": 0,
                    "policy_init": 0,
                    "grounding": 50000,
                    "supervised_training": 10,
                    "policy_training": 1e5,
                }
            )
        if args.noisy_baseline:
            parser.set_defaults(
                max_stage_steps={
                    "policy_init": 5e5,
                    "grounding": 0,
                    "policy_training": 0,
                }
            )
    elif "Push" in args.source_env:
        parser.set_defaults(policy_training_evaluate_interval=1000)
        parser.set_defaults(
            max_stage_steps={
                "supervised_pretraining": 80,
                "policy_init": 1e5,
                "grounding": 50000,
                "supervised_training": 10,
                "policy_training": 1e4,
            }
        )
        if args.run_prefix == "more_grounding":
            parser.set_defaults(
                max_stage_steps={
                    "supervised_pretraining": 0,
                    "policy_init": 0,
                    "grounding": 50000,
                    "supervised_training": 10,
                    "policy_training": 1e4,
                }
            )
        if args.run_prefix == "x_visual_norecon":
            parser.set_defaults(
                max_stage_steps={
                    "supervised_pretraining": 40,
                    "policy_init": 1e5,
                    "grounding": 50000,
                    "supervised_training": 5,
                    "policy_training": 1e4,
                }
            )
        if args.noisy_baseline:
            parser.set_defaults(
                max_stage_steps={
                    "policy_init": 1e5,
                    "grounding": 0,
                    "policy_training": 0,
                }
            )
    elif "InvertedPendulum" in args.source_env:
        parser.set_defaults(policy_training_evaluate_interval=500)
        parser.set_defaults(
            max_stage_steps={
                "supervised_pretraining": 80,
                "policy_init": 1e5,
                "grounding": 50000,
                "supervised_training": 10,
                "policy_training": 1e4,
            }
        )
        if args.noisy_baseline:
            parser.set_defaults(
                max_stage_steps={
                    "policy_init": 5e4,
                    "grounding": 0,
                    "policy_training": 0,
                }
            )
    elif "Fetch" in args.source_env:
        parser.set_defaults(policy_training_evaluate_interval=1000)
        parser.set_defaults(
            max_stage_steps={
                "supervised_pretraining": 80,
                "policy_init": 5e5,
                "grounding": 50000,
                "supervised_training": 10,
                "policy_training": 1e4,
            }
        )
        if args.noisy_baseline:
            parser.set_defaults(
                max_stage_steps={
                    "policy_init": 1e5,
                    "grounding": 0,
                    "policy_training": 0,
                }
            )

    if args.run_prefix == "video" or args.run_prefix == "eval":
        parser.set_defaults(
            max_stage_steps={"policy_init": 0, "grounding": 0, "policy_training": 0}
        )

    if args.encoder_type == "cnn" and not args.run_rl_from_state:
        parser.set_defaults(gaifo_discriminator_mlp_dim=[1024, 1024])
        parser.set_defaults(gaifo_discriminator_batch_size=128)
        parser.set_defaults(policy_mlp_dim=[1024, 1024])
        parser.set_defaults(critic_mlp_dim=[1024, 1024])
        parser.set_defaults(discriminator_mlp_dim=[1024, 1024])
        parser.set_defaults(screen_width=100)
        parser.set_defaults(screen_height=100)
        parser.set_defaults(encoder_image_size=92)

    if args.encoder_type != "cnn" or args.run_rl_from_state:
        parser.set_defaults(discriminator_mlp_dim=[64, 64])
        parser.set_defaults(policy_mlp_dim=[64, 64])
        parser.set_defaults(critic_mlp_dim=[64, 64])
