DEFAULT_MODELS_OPEN_MAGVIT2_IMAGE = {
    "TencentARC/Open-MAGVIT2-Tokenizer-128-resolution",
    "TencentARC/Open-MAGVIT2-Tokenizer-256-resolution",
    "TencentARC/Open-MAGVIT2-Tokenizer-262144-Pretrain",
    "TencentARC/Open-MAGVIT2-Tokenizer-16384-Pretrain"
}

CONFIGS_OPEN_MAGVIT2_IMAGE = {
    "256-GPU": {
        "seed_everything": True,
        "trainer": {
            "accelerator": "gpu",
            "strategy": "ddp_find_unused_parameters_true",
            "devices": 8,
            "num_nodes": 4,
            "precision": "16-mixed",
            "max_epochs": 270,
            "check_val_every_n_epoch": 1,
            "num_sanity_val_steps": -1,
            "log_every_n_steps": 100,
            "callbacks": [
                {
                    "class_path": "lightning.pytorch.callbacks.ModelCheckpoint",
                    "init_args": {
                        "dirpath": "../../checkpoints/vqgan/test",
                        "save_top_k": -1  # save all checkpoints
                    }
                },
                {
                    "class_path": "lightning.pytorch.callbacks.LearningRateMonitor",
                    "init_args": {
                        "logging_interval": "step"
                    }
                }
            ],
            "logger": {
                "class_path": "lightning.pytorch.loggers.TensorBoardLogger",
                "init_args": {
                    "save_dir": "../../results/vqgan/",
                    "version": "test",
                    "name": ""
                }
            }
        },
        "model": {
            "class_path": "OpenImageTokenizer.Open_MAGVIT2.models.lfqgan.VQModel",
            "init_args": {
                "ddconfig": {
                    "double_z": False,
                    "z_channels": 18,
                    "resolution": 128,
                    "in_channels": 3,
                    "out_ch": 3,
                    "ch": 128,
                    "ch_mult": [1, 1, 2, 2, 4],  # num_down = len(ch_mult)-1
                    "num_res_blocks": 4
                },
                "lossconfig": {
                    "target": "OpenImageTokenizer.Open_MAGVIT2.modules.losses.vqperceptual.VQLPIPSWithDiscriminator",
                    "params": {
                        "disc_conditional": False,
                        "disc_in_channels": 3,
                        "disc_start": 0,  # from 0 epoch
                        "disc_weight": 0.8,
                        "gen_loss_weight": 0.1,
                        "lecam_loss_weight": 0.05,
                        "codebook_weight": 0.1,
                        "commit_weight": 0.25,
                        "codebook_enlarge_ratio": 0,
                        "codebook_enlarge_steps": 2000
                    }
                },
                "n_embed": 262144,
                "embed_dim": 18,
                "learning_rate": 1e-4,
                "sample_minimization_weight": 1.0,
                "batch_maximization_weight": 1.0,
                "scheduler_type": "None",
                "use_ema": True,
                "resume_lr": None,
                "lr_drop_epoch": [200, 250]
            }
        },
        "data": {
            "class_path": "main.DataModuleFromConfig",
            "init_args": {
                "batch_size": 8,
                "num_workers": 16,
                "train": {
                    "target": "OpenImageTokenizer.Open_MAGVIT2.data.imagenet.ImageNetTrain",
                    "params": {
                        "config": {
                            "size": 256,
                            "subset": None
                        }
                    }
                },
                "validation": {
                    "target": "OpenImageTokenizer.Open_MAGVIT2.data.imagenet.ImageNetValidation",
                    "params": {
                        "config": {
                            "size": 256,
                            "subset": None
                        }
                    }
                },
                "test": {
                    "target": "OpenImageTokenizer.Open_MAGVIT2.data.imagenet.ImageNetValidation",
                    "params": {
                        "config": {
                            "size": 256,
                            "subset": None
                        }
                    }
                }
            }
        },
        "ckpt_path": None  # to resume
    },
    "pretrain-16384" : {
        "seed_everything": True,
        "trainer": {
            "accelerator": "gpu",
            "strategy" : "ddp_find_unused_parameters_true",
            "devices": 8,
            "num_nodes": 4,
            "precision": "16-mixed",
            "max_epochs": 1500000,
            "check_val_every_n_epoch": None,
            "val_check_interval": 5005,
            "num_sanity_val_steps": -1,
            "log_every_n_steps": 100,
            "callbacks" : [
                {
                    "class_path" : "lightning.pytorch.callbacks.ModelCheckpoint",
                    "init_args" : {
                        "dirpath": "../../checkpoints/vqgan/test",
                        "save_top_k": -1  # save all checkpoints
                    }
                },
                {
                    "class_path": "lightning.pytorch.callbacks.LearningRateMonitor",
                    "init_args": {
                        "logging_interval": "step"
                    }
                }
            ],
            "logger" : {
                "class_path" : "lightning.pytorch.loggers.TensorBoardLogger",
                "init_args" : {
                    "save_dir": "../../results/vqgan/",
                    "version": "test",
                    "name": ""
                }
            }
        },
        "model" : {
            "class_path" : "OpenImageTokenizer.Open_MAGVIT2.models.lfqgan_pretrain.VQModel",
            "init_args" : {
                "ddconfig": {
                    "double_z": False,
                    "z_channels": 14,
                    "resolution": 128,
                    "in_channels": 3,
                    "out_ch": 3,
                    "ch": 128,
                    "ch_mult": [1,1,2,2,4],  # num_down = len(ch_mult)-1
                    "num_res_blocks": 4
                },
                "lossconfig" : {
                    "target" : "OpenImageTokenizer.Open_MAGVIT2.modules.losses.vqperceptual.VQLPIPSWithDiscriminator",
                    "params" : {
                        "disc_conditional": False,
                        "disc_in_channels": 3,
                        "disc_start": 0,  # from 0 epoch
                        "disc_num_layers": 4,
                        "disc_weight": 0.8,
                        "gen_loss_weight": 0.1,
                        "lecam_loss_weight": 0.05,
                        "codebook_weight": 0.1,
                        "commit_weight": 0.25,
                        "codebook_enlarge_ratio": 0,
                        "codebook_enlarge_steps": 2000,
                        "disc_loss" : "hinge",
                        "disc_num_channels" : 3,
                        "disc_num_stages" : 3,
                        "disc_hidden_channels" : 128,
                        "blur_resample" : True,
                        "blur_kernel_size" : 4,
                        "use_blur" : True
                    }
                },
                "n_embed": 16384,
                "embed_dim": 14,
                "learning_rate": 1e-4,
                "sample_minimization_weight": 1.0,
                "batch_maximization_weight": 1.0,
                "scheduler_type": "None",
                "use_ema": True,
                "use_shared_epoch": True,
                "sche_type" : "cos",
                "wpe" : 0.01,
                "wp" : 1,
                "wp0" : 0.0,
                "max_iter" : 1500000,
                "wp_iter" : 5000
            }
        },
        "data" : {
            "class_path" : "main.DataModuleFromConfig",
            "init_args" : {
                "batch_size": 8,
                "num_workers": 16,
                "train": {
                    "target": "OpenImageTokenizer.Open_MAGVIT2.data.pretrain.LAIONCombineTrain",
                    "params": {
                        "config": {
                            "size": 256,
                            "subset": None,
                            "filter_path" : ["../../data/laion-aesthetic-v2_filter_keys.json", "../../data/JourneyDB_filter_keys.json", "../../data/laion-aesthetic_v1_filter_keys.json", "../../data/laion-hd_sub_filter_keys_2.json"],
                            "sample_json_path": ["../../data/laion-coco_samples.json", "../../data/cc15m_samples_2.json", "../../data/laion-aesthetic-v2_samples.json", "../../data/JourneyDB_samples.json", "../../data/laion-aesthetic_v1_samples.json", "../../data/laion-hd_sub_samples_2.json"],
                            "sample_coco_urls": "../../data/laion-coco_sample_urls_20M.txt #please specify your path",
                            "sample_hd_urls" : "../../data/laion-hd_sample_urls_30M_2.txt ##please specify your path",
                            "data_dir" : ["../../data/LAION-COCO-Recaption", "../../data/CC12M/webdataset/gcc12m_shards", "../../data/Laion-aesthetic-v2/data", "../../data/CC3M/webdataset/gcc3m_shards", "../../data/public_datasets/JourneyDB/wds", "../../data/laion-aesthetics-12M/webdataset_train", "../../public_datasets/laion-hd/webdataset_train/"],
                            "image_key" : ["jpg", "jpeg.jpg", "jpg.jpg"],
                            "enable_image" : True
                        }
                    }
                },
                "validation": {
                    "target": "OpenImageTokenizer.Open_MAGVIT2.data.imagenet.ImageNetValidation",
                    "params": {
                        "config": {
                            "size": 256,
                            "subset": None
                        }
                    }
                },
                "test": {
                    "target": "OpenImageTokenizer.Open_MAGVIT2.data.imagenet.ImageNetValidation",
                    "params": {
                        "config": {
                            "size": 256,
                            "subset": None
                        }
                    }
                }
            }
        },
        "ckpt_path": None  # to resume
    },
    "pretrain-262144" : {
        "seed_everything": True,
        "trainer": {
            "accelerator": "gpu",
            "strategy" : "ddp_find_unused_parameters_true",
            "devices": 8,
            "num_nodes": 4,
            "precision": "16-mixed",
            "max_epochs": 1500000,
            "check_val_every_n_epoch": None,
            "val_check_interval": 5005,
            "num_sanity_val_steps": -1,
            "log_every_n_steps": 100,
            "callbacks" : [
                {
                    "class_path" : "lightning.pytorch.callbacks.ModelCheckpoint",
                    "init_args" : {
                        "dirpath": "../../checkpoints/vqgan/test",
                        "save_top_k": -1  # save all checkpoints
                    }
                },
                {
                    "class_path": "lightning.pytorch.callbacks.LearningRateMonitor",
                    "init_args": {
                        "logging_interval": "step"
                    }
                }
            ],
            "logger" : {
                "class_path" : "lightning.pytorch.loggers.TensorBoardLogger",
                "init_args" : {
                    "save_dir": "../../results/vqgan/",
                    "version": "test",
                    "name": ""
                }
            }
        },
        "model" : {
            "class_path" : "OpenImageTokenizer.Open_MAGVIT2.models.lfqgan_pretrain.VQModel",
            "init_args" : {
                "ddconfig": {
                    "double_z": False,
                    "z_channels": 18,
                    "resolution": 128,
                    "in_channels": 3,
                    "out_ch": 3,
                    "ch": 128,
                    "ch_mult": [1,1,2,2,4],  # num_down = len(ch_mult)-1
                    "num_res_blocks": 4
                },
                "lossconfig" : {
                    "target" : "OpenImageTokenizer.Open_MAGVIT2.modules.losses.vqperceptual.VQLPIPSWithDiscriminator",
                    "params" : {
                        "disc_conditional": False,
                        "disc_in_channels": 3,
                        "disc_start": 0,  # from 0 epoch
                        "disc_num_layers": 3,
                        "disc_weight": 0.8,
                        "gen_loss_weight": 0.1,
                        "lecam_loss_weight": 0.05,
                        "codebook_weight": 0.1,
                        "commit_weight": 0.25,
                        "codebook_enlarge_ratio": 0,
                        "codebook_enlarge_steps": 2000,
                        "disc_loss" : "hinge",
                        "disc_num_channels" : 3,
                        "disc_num_stages" : 3,
                        "disc_hidden_channels" : 128,
                        "blur_resample" : True,
                        "blur_kernel_size" : 4
                    }
                },
                "n_embed": 262144,
                "embed_dim": 18,
                "learning_rate": 1e-4,
                "sample_minimization_weight": 1.0,
                "batch_maximization_weight": 1.0,
                "scheduler_type": "None",
                "use_ema": True,
                "use_shared_epoch": True,
                "sche_type" : "",
                "wpe" : 0.01,
                "wp" : 1,
                "wp0" : 0.0,
                "max_iter" : 1500000,
                "wp_iter" : 5000,
                "lr_drop_iter" : [800000, 1000000]
            }
        },
        "data" : {
            "class_path" : "main.DataModuleFromConfig",
            "init_args" : {
                "batch_size": 8,
                "num_workers": 16,
                "train": {
                    "target": "OpenImageTokenizer.Open_MAGVIT2.data.pretrain.LAIONCombineTrain",
                    "params": {
                        "config": {
                            "size": 256,
                            "subset": None,
                            "filter_path" : ["../../data/laion-aesthetic-v2_filter_keys.json", "../../data/JourneyDB_filter_keys.json", "../../data/laion-aesthetic_v1_filter_keys.json", "../../data/laion-hd_sub_filter_keys_2.json"],
                            "sample_json_path": ["../../data/laion-coco_samples.json", "../../data/cc15m_samples_2.json", "../../data/laion-aesthetic-v2_samples.json", "../../data/JourneyDB_samples.json", "../../data/laion-aesthetic_v1_samples.json", "../../data/laion-hd_sub_samples_2.json"],
                            "sample_coco_urls": "../../data/laion-coco_sample_urls_20M.txt #please specify your path",
                            "sample_hd_urls" : "../../data/laion-hd_sample_urls_30M_2.txt ##please specify your path",
                            "data_dir" : ["../../data/LAION-COCO-Recaption", "../../data/CC12M/webdataset/gcc12m_shards", "../../data/Laion-aesthetic-v2/data", "../../data/CC3M/webdataset/gcc3m_shards", "../../data/public_datasets/JourneyDB/wds", "../../data/laion-aesthetics-12M/webdataset_train", "../../public_datasets/laion-hd/webdataset_train/"],
                            "image_key" : ["jpg", "jpeg.jpg", "jpg.jpg"],
                            "enable_image" : True
                        }
                    }
                },
                "validation": {
                    "target": "OpenImageTokenizer.Open_MAGVIT2.data.imagenet.ImageNetValidation",
                    "params": {
                        "config": {
                            "size": 256,
                            "subset": None
                        }
                    }
                },
                "test": {
                    "target": "OpenImageTokenizer.Open_MAGVIT2.data.imagenet.ImageNetValidation",
                    "params": {
                        "config": {
                            "size": 256,
                            "subset": None
                        }
                    }
                }
            }
        },
        "ckpt_path": None  # to resume
    },
    "128-GPU" : {
        "seed_everything": True,
        "trainer": {
            "accelerator": "gpu",
            "strategy" : "ddp_find_unused_parameters_true",
            "devices": 8,
            "num_nodes": 4,
            "precision": "16-mixed",
            "max_epochs": 350,
            "check_val_every_n_epoch": 1,
            "num_sanity_val_steps": -1,
            "log_every_n_steps": 100,
            "callbacks" : [
                {
                    "class_path" : "lightning.pytorch.callbacks.ModelCheckpoint",
                    "init_args" : {
                        "dirpath": "../../checkpoints/vqgan/test",
                        "save_top_k": -1  # save all checkpoints
                    }
                },
                {
                    "class_path": "lightning.pytorch.callbacks.LearningRateMonitor",
                    "init_args": {
                        "logging_interval": "step"
                    }
                }
            ],
            "logger" : {
                "class_path" : "lightning.pytorch.loggers.TensorBoardLogger",
                "init_args" : {
                    "save_dir": "../../results/vqgan/",
                    "version": "test",
                    "name": ""
                }
            }
        },
        "model" : {
            "class_path" : "OpenImageTokenizer.Open_MAGVIT2.models.lfqgan.VQModel",
            "init_args" : {
                "ddconfig": {
                    "double_z": False,
                    "z_channels": 18,
                    "resolution": 128,
                    "in_channels": 3,
                    "out_ch": 3,
                    "ch": 128,
                    "ch_mult": [1,2,2,4],  # num_down = len(ch_mult)-1
                    "num_res_blocks": 4
                },
                "lossconfig" : {
                    "target" : "OpenImageTokenizer.Open_MAGVIT2.modules.losses.vqperceptual.VQLPIPSWithDiscriminator",
                    "params" : {
                        "disc_conditional": False,
                        "disc_in_channels": 3,
                        "disc_start": 0,  # from 0 epoch
                        "disc_weight": 0.8,
                        "gen_loss_weight": 0.1,
                        "lecam_loss_weight": 0.01,
                        "codebook_weight": 0.1,
                        "commit_weight": 0.25,
                        "codebook_enlarge_ratio": 0,
                        "codebook_enlarge_steps": 2000
                    }
                },
                "n_embed": 262144,
                "embed_dim": 18,
                "learning_rate": 1e-4,
                "sample_minimization_weight": 1.0,
                "batch_maximization_weight": 1.0,
                "scheduler_type": "None",
                "use_ema": True,
                "resume_lr": None,
                "lr_drop_epoch": [250, 300]
            }
        },
        "data" : {
            "class_path" : "main.DataModuleFromConfig",
            "init_args" : {
                "batch_size": 8,
                "num_workers": 16,
                "train": {
                    "target": "OpenImageTokenizer.Open_MAGVIT2.data.imagenet.ImageNetTrain",
                    "params": {
                        "config": {
                            "size": 128,
                            "subset": None
                        }
                    }
                },
                "validation": {
                    "target": "OpenImageTokenizer.Open_MAGVIT2.data.imagenet.ImageNetValidation",
                    "params": {
                        "config": {
                            "size": 128,
                            "subset": None
                        }
                    }
                },
                "test": {
                    "target": "OpenImageTokenizer.Open_MAGVIT2.data.imagenet.ImageNetValidation",
                    "params": {
                        "config": {
                            "size": 128,
                            "subset": None
                        }
                    }
                }
            }
        },
        "ckpt_path": None  # to resume
    }
}

MODEL_TO_CONFIG = {
    "TencentARC/Open-MAGVIT2-Tokenizer-128-resolution": "128-GPU",
    "TencentARC/Open-MAGVIT2-Tokenizer-256-resolution": "256-GPU",
    "TencentARC/Open-MAGVIT2-Tokenizer-16384-Pretrain": "pretrain-16384",
    "TencentARC/Open-MAGVIT2-Tokenizer-262144-Pretrain": "pretrain-262144",
    "TencentARC/IBQ-Tokenizer-1024": "imagenet256_1024",
    "TencentARC/IBQ-Tokenizer-8192": "imagenet256_8192",
    "TencentARC/IBQ-Tokenizer-16384": "imagenet256_16384",
    "TencentARC/IBQ-Tokenizer-262144": "imagenet256_262144"
}

CONFIGS_IBQ_IMAGE = {
    "imagenet256_1024" : {
        "seed_everything": True,
        "trainer": {
            "accelerator": "gpu",
            "strategy" : "ddp_find_unused_parameters_true",
            "devices": 8,
            "num_nodes": 8,
            "precision": "16-mixed",
            "max_epochs": 330,
            "check_val_every_n_epoch": 1,
            "num_sanity_val_steps": -1,
            "callbacks" : [
                {
                    "class_path" : "lightning.pytorch.callbacks.ModelCheckpoint",
                    "init_args" : {
                        "dirpath": "../../checkpoints/vqgan/test",
                        "save_top_k": -1,  # save all checkpoints
                        "save_last": True,
                        "monitor": "train/perceptual_loss"
                    }
                },
                {
                    "class_path": "lightning.pytorch.callbacks.LearningRateMonitor",
                    "init_args": {
                        "logging_interval": "step"
                    }
                }
            ],
            "logger" : {
                "class_path" : "lightning.pytorch.loggers.TensorBoardLogger",
                "init_args" : {
                    "save_dir": "../../results/vqgan/",
                    "version": "test",
                    "name": ""
                }
            }
        },
        "model" : {
            "class_path" : "OpenImageTokenizer.IBQ.models.ibqgan.IBQ",
            "init_args" : {
                "ddconfig": {
                    "double_z": False,
                    "z_channels": 256,
                    "resolution": 256,
                    "in_channels": 3,
                    "out_ch": 3,
                    "ch": 128,
                    "ch_mult": [1,1,2,2,4],  # num_down = len(ch_mult)-1
                    "num_res_blocks": 4,
                    "attn_resolutions": [16],
                    "dropout": 0.0
                },
                "lossconfig" : {
                    "target" : "OpenImageTokenizer.IBQ.modules.losses.vqperceptual.VQLPIPSWithDiscriminator",
                    "params" : {
                        "disc_conditional": False,
                        "disc_in_channels": 3,
                        "disc_start": 0,  # from 0 epoch
                        "disc_weight": 0.4,
                        "quant_loss_weight": 1.0,
                        "entropy_loss_weight": 0.05,
                        "gen_loss_weight": 0.1,
                        "lecam_loss_weight": 0.01
                    }
                },
                "n_embed": 1024,
                "embed_dim": 256,
                "learning_rate": 1e-4,
                "l2_normalize": False,
                "use_entropy_loss": True,
                "sample_minimization_weight": 1.0,
                "batch_maximization_weight": 1.0,
                "use_ema": True,
                "resume_lr": None,
                "lr_drop_epoch": [250, 300]
            }
        },
        "data" : {
            "class_path" : "main.DataModuleFromConfig",
            "init_args" : {
                "batch_size": 4,
                "num_workers": 16,
                "train": {
                    "target": "OpenImageTokenizer.IBQ.data.imagenet.ImageNetTrain",
                    "params": {
                        "config": {
                            "size": 256,
                            "subset": None
                        }
                    }
                },
                "validation": {
                    "target": "OpenImageTokenizer.IBQ.data.imagenet.ImageNetValidation",
                    "params": {
                        "config": {
                            "size": 256,
                            "subset": None
                        }
                    }
                },
                "test": {
                    "target": "OpenImageTokenizer.IBQ.data.imagenet.ImageNetValidation",
                    "params": {
                        "config": {
                            "size": 256,
                            "subset": None
                        }
                    }
                }
            }
        },
        "ckpt_path": None  # to resume
    },
    "imagenet256_8192" : {
        "seed_everything": True,
        "trainer": {
            "accelerator": "gpu",
            "strategy" : "ddp_find_unused_parameters_true",
            "devices": 8,
            "num_nodes": 8,
            "precision": "16-mixed",
            "max_epochs": 280,
            "check_val_every_n_epoch": 1,
            "num_sanity_val_steps": -1,
            "callbacks" : [
                {
                    "class_path" : "lightning.pytorch.callbacks.ModelCheckpoint",
                    "init_args" : {
                        "dirpath": "../../checkpoints/vqgan/test",
                        "save_top_k": -1,  # save all checkpoints
                        "save_last": True,
                        "monitor": "train/perceptual_loss"
                    }
                },
                {
                    "class_path": "lightning.pytorch.callbacks.LearningRateMonitor",
                    "init_args": {
                        "logging_interval": "step"
                    }
                }
            ],
            "logger" : {
                "class_path" : "lightning.pytorch.loggers.TensorBoardLogger",
                "init_args" : {
                    "save_dir": "../../results/vqgan/",
                    "version": "test",
                    "name": ""
                }
            }
        },
        "model" : {
            "class_path" : "OpenImageTokenizer.IBQ.models.ibqgan.IBQ",
            "init_args" : {
                "ddconfig": {
                    "double_z": False,
                    "z_channels": 256,
                    "resolution": 256,
                    "in_channels": 3,
                    "out_ch": 3,
                    "ch": 128,
                    "ch_mult": [1,1,2,2,4],  # num_down = len(ch_mult)-1
                    "num_res_blocks": 4,
                    "attn_resolutions": [16],
                    "dropout": 0.0
                },
                "lossconfig" : {
                    "target" : "OpenImageTokenizer.IBQ.modules.losses.vqperceptual.VQLPIPSWithDiscriminator",
                    "params" : {
                        "disc_conditional": False,
                        "disc_in_channels": 3,
                        "disc_start": 0,  # from 0 epoch
                        "disc_weight": 0.4,
                        "quant_loss_weight": 1.0,
                        "entropy_loss_weight": 0.05,
                        "gen_loss_weight": 0.1,
                        "lecam_loss_weight": 0.05
                    }
                },
                "n_embed": 8192,
                "embed_dim": 256,
                "learning_rate": 1e-4,
                "l2_normalize": False,
                "use_entropy_loss": True,
                "sample_minimization_weight": 1.0,
                "batch_maximization_weight": 1.0,
                "entropy_temperature": 0.01,
                "beta": 0.25,
                "use_ema": True,
                "resume_lr": None,
                "lr_drop_epoch": [250]
            }
        },
        "data" : {
            "class_path" : "main.DataModuleFromConfig",
            "init_args" : {
                "batch_size": 4,
                "num_workers": 16,
                "train": {
                    "target": "OpenImageTokenizer.IBQ.data.imagenet.ImageNetTrain",
                    "params": {
                        "config": {
                            "size": 256,
                            "subset": None
                        }
                    }
                },
                "validation": {
                    "target": "OpenImageTokenizer.IBQ.data.imagenet.ImageNetValidation",
                    "params": {
                        "config": {
                            "size": 256
                        }
                    }
                },
                "test": {
                    "target": "OpenImageTokenizer.IBQ.data.imagenet.ImageNetValidation",
                    "params": {
                        "config": {
                            "size": 256
                        }
                    }
                }
            }
        },
        "ckpt_path": None  # to resume
    },
    "imagenet256_16384" : {
        "seed_everything": True,
        "trainer": {
            "accelerator": "gpu",
            "strategy" : "ddp_find_unused_parameters_true",
            "devices": 8,
            "num_nodes": 8,
            "precision": "16-mixed",
            "max_epochs": 330,
            "check_val_every_n_epoch": 1,
            "num_sanity_val_steps": -1,
            "callbacks" : [
                {
                    "class_path" : "lightning.pytorch.callbacks.ModelCheckpoint",
                    "init_args" : {
                        "dirpath": "../../checkpoints/vqgan/test",
                        "save_top_k": -1,  # save all checkpoints
                        "save_last": True,
                        "monitor": "train/perceptual_loss"
                    }
                },
                {
                    "class_path": "lightning.pytorch.callbacks.LearningRateMonitor",
                    "init_args": {
                        "logging_interval": "step"
                    }
                }
            ],
            "logger" : {
                "class_path" : "lightning.pytorch.loggers.TensorBoardLogger",
                "init_args" : {
                    "save_dir": "../../results/vqgan/",
                    "version": "test",
                    "name": ""
                }
            }
        },
        "model" : {
            "class_path" : "OpenImageTokenizer.IBQ.models.ibqgan.IBQ",
            "init_args" : {
                "ddconfig": {
                    "double_z": False,
                    "z_channels": 256,
                    "resolution": 256,
                    "in_channels": 3,
                    "out_ch": 3,
                    "ch": 128,
                    "ch_mult": [1,1,2,2,4],  # num_down = len(ch_mult)-1
                    "num_res_blocks": 4,
                    "attn_resolutions": [16],
                    "dropout": 0.0
                },
                "lossconfig" : {
                    "target" : "OpenImageTokenizer.IBQ.modules.losses.vqperceptual.VQLPIPSWithDiscriminator",
                    "params" : {
                        "disc_conditional": False,
                        "disc_in_channels": 3,
                        "disc_start": 0,  # from 0 epoch
                        "disc_weight": 0.4,
                        "quant_loss_weight": 1.0,
                        "entropy_loss_weight": 0.05,
                        "gen_loss_weight": 0.1,
                        "lecam_loss_weight": 0.05
                    }
                },
                "n_embed": 16384,
                "embed_dim": 256,
                "learning_rate": 1e-4,
                "l2_normalize": False,
                "use_entropy_loss": True,
                "sample_minimization_weight": 1.0,
                "batch_maximization_weight": 1.0,
                "entropy_temperature": 0.01,
                "beta": 0.25,
                "use_ema": True,
                "resume_lr": None,
                "lr_drop_epoch": [250, 330]
            }
        },
        "data" : {
            "class_path" : "main.DataModuleFromConfig",
            "init_args" : {
                "batch_size": 4,
                "num_workers": 16,
                "train": {
                    "target": "OpenImageTokenizer.IBQ.data.imagenet.ImageNetTrain",
                    "params": {
                        "config": {
                            "size": 256,
                            "subset": None
                        }
                    }
                },
                "validation": {
                    "target": "OpenImageTokenizer.IBQ.data.imagenet.ImageNetValidation",
                    "params": {
                        "config": {
                            "size": 256
                        }
                    }
                },
                "test": {
                    "target": "OpenImageTokenizer.IBQ.data.imagenet.ImageNetValidation",
                    "params": {
                        "config": {
                            "size": 256
                        }
                    }
                }
            }
        },
        "ckpt_path": None  # to resume
    },
    "imagenet256_262144" : {
        "seed_everything": True,
        "trainer": {
            "accelerator": "gpu",
            "strategy" : "ddp_find_unused_parameters_true",
            "devices": 8,
            "num_nodes": 8,
            "precision": "16-mixed",
            "max_epochs": 330,
            "check_val_every_n_epoch": 1,
            "num_sanity_val_steps": -1,
            "callbacks" : [
                {
                    "class_path" : "lightning.pytorch.callbacks.ModelCheckpoint",
                    "init_args" : {
                        "dirpath": "../../checkpoints/vqgan/test",
                        "save_top_k": -1,  # save all checkpoints
                        "save_last": True,
                        "monitor": "train/perceptual_loss"
                    }
                },
                {
                    "class_path": "lightning.pytorch.callbacks.LearningRateMonitor",
                    "init_args": {
                        "logging_interval": "step"
                    }
                }
            ],
            "logger" : {
                "class_path" : "lightning.pytorch.loggers.TensorBoardLogger",
                "init_args" : {
                    "save_dir": "../../results/vqgan/",
                    "version": "test",
                    "name": ""
                }
            }
        },
        "model" : {
            "class_path" : "OpenImageTokenizer.IBQ.models.ibqgan.IBQ",
            "init_args" : {
                "ddconfig": {
                    "double_z": False,
                    "z_channels": 256,
                    "resolution": 256,
                    "in_channels": 3,
                    "out_ch": 3,
                    "ch": 128,
                    "ch_mult": [1,1,2,2,4],  # num_down = len(ch_mult)-1
                    "num_res_blocks": 4,
                    "attn_resolutions": [16],
                    "dropout": 0.0
                },
                "lossconfig" : {
                    "target" : "OpenImageTokenizer.IBQ.modules.losses.vqperceptual.VQLPIPSWithDiscriminator",
                    "params" : {
                        "disc_conditional": False,
                        "disc_in_channels": 3,
                        "disc_start": 0,  # from 0 epoch
                        "disc_weight": 0.4,
                        "quant_loss_weight": 1.0,
                        "entropy_loss_weight": 0.05,
                        "gen_loss_weight": 0.1,
                        "lecam_loss_weight": 0.05
                    }
                },
                "n_embed": 262144,
                "embed_dim": 256,
                "learning_rate": 1e-4,
                "l2_normalize": False,
                "use_entropy_loss": True,
                "sample_minimization_weight": 1.0,
                "batch_maximization_weight": 1.0,
                "entropy_temperature": 0.01,
                "beta": 0.25,
                "use_ema": True,
                "resume_lr": None,
                "lr_drop_epoch": [250, 330]
            }
        },
        "data" : {
            "class_path" : "main.DataModuleFromConfig",
            "init_args" : {
                "batch_size": 4,
                "num_workers": 16,
                "train": {
                    "target": "OpenImageTokenizer.IBQ.data.imagenet.ImageNetTrain",
                    "params": {
                        "config": {
                            "size": 256,
                            "subset": None
                        }
                    }
                },
                "validation": {
                    "target": "OpenImageTokenizer.IBQ.data.imagenet.ImageNetValidation",
                    "params": {
                        "config": {
                            "size": 256
                        }
                    }
                },
                "test": {
                    "target": "OpenImageTokenizer.IBQ.data.imagenet.ImageNetValidation",
                    "params": {
                        "config": {
                            "size": 256
                        }
                    }
                }
            }
        },
        "ckpt_path": None  # to resume
    }
}

def get_model_config(model_name):

    if model_name not in MODEL_TO_CONFIG:
        raise ValueError(f"Modelo no reconocido: {model_name}")

    config_key = MODEL_TO_CONFIG[model_name]

    if config_key not in CONFIGS_OPEN_MAGVIT2_IMAGE:
        raise ValueError(f"Configuración no encontrada: {config_key}")
    
    return CONFIGS_OPEN_MAGVIT2_IMAGE[config_key]

def get_model_config_IBQ(model_name):
    if model_name not in MODEL_TO_CONFIG:
        raise ValueError(f"Modelo no reconocido: {model_name}")

    config_key = MODEL_TO_CONFIG[model_name]

    if config_key not in CONFIGS_IBQ_IMAGE:
        raise ValueError(f"Configuración no encontrada: {config_key}")
    
    return CONFIGS_IBQ_IMAGE[config_key]
