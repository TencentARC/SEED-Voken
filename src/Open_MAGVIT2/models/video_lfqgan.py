import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import lightning as L

from main import instantiate_from_config
from contextlib import contextmanager

from src.Open_MAGVIT2.modules.diffusionmodules.improved_video_model import Encoder, Decoder
from src.Open_MAGVIT2.modules.vqvae.lookup_free_quantize import LFQ
from src.Open_MAGVIT2.modules.scheduler.lr_scheduler import Scheduler_LinearWarmup, Scheduler_LinearWarmup_CosineDecay
from src.Open_MAGVIT2.modules.util import requires_grad
from src.Open_MAGVIT2.modules.ema import LitEma

import math
from collections import OrderedDict

TARGET_RESOLUTION = (224, 224)

class VQModel(L.LightningModule):
    def __init__(self,
                ddconfig,
                lossconfig,
                n_embed,
                embed_dim,
                sample_minimization_weight,
                batch_maximization_weight,
                ckpt_path=None,
                ignore_keys=[],
                image_key="video",
                colorize_nlabels=None,
                monitor=None,
                learning_rate=None,
                ### scheduler config
                warmup_epochs=1.0, #warmup epochs
                scheduler_type = "linear-warmup_cosine-decay",
                min_learning_rate = 0,
                use_ema = True,
                stage = None,
                use_shared_epoch = False,
                image_pretrain_path = None,
                wp = 0,
                wp0 = 0.005, #initial lr ratio at the begging of lr warm up
                wpe = 0.01, #final lr ratio at the end of training
                max_iter = None,
                sche_type = None,
                wp_iter = None,
                resume_lr = None,
                ):
        super().__init__()
        self.image_key = image_key
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.loss = instantiate_from_config(lossconfig)
        self.quantize = LFQ(dim=embed_dim, codebook_size=n_embed, sample_minimization_weight=sample_minimization_weight, batch_maximization_weight=batch_maximization_weight)
        
        self.image_key = image_key
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor

        self.use_ema = use_ema
        self.use_shared_epoch = use_shared_epoch
        
        if self.use_ema and stage is None:
            self.model_ema = LitEma(self)
        if image_pretrain_path is not None: ##perform inflation
            self.inflate_from_image(image_pretrain_path)
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

        self.learning_rate = learning_rate
        self.scheduler_type = scheduler_type
        self.warmup_epochs = warmup_epochs
        self.min_learning_rate = min_learning_rate
        self.automatic_optimization = False

        ## scheduler related
        self.wp = wp
        self.wp0 = wp0
        self.wpe = wpe
        self.sche_type = sche_type
        self.max_it = max_iter
        self.wp_iter = wp_iter
        self.resume_lr = resume_lr

        self.strict_loading = False

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.parameters())
            self.model_ema.copy_to(self)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def inflate_from_image(self, image_pretrain_path):
        """
        use last temporal inflation as MAGVIT2 does
        """
        assert image_pretrain_path is not None
        image_pretrain_sd = torch.load(image_pretrain_path, map_location="cpu")["state_dict"]
        new_video_encoder_params = self.encoder.state_dict()
        new_video_decoder_params = self.decoder.state_dict()
        original_image_encoder_params = OrderedDict()
        original_image_decoder_params = OrderedDict()
        original_ema_params = OrderedDict()
        ema_mapping_temp = dict()
        image_ema_mapping = dict()
        ## load original parameters
        ##first extract the encoder and decoder weight
        for key, value in image_pretrain_sd.items():
            if "model_ema." not in key: ## load original param not load D
                if "encoder" in key:
                    original_image_encoder_params[key] = value
                elif "decoder" in key:
                    original_image_decoder_params[key] = value
                ema_key = key.replace(".", "")
                ema_mapping_temp[ema_key] = key
            else:
                if "discriminator" in key or "num_updates" in key or "decay" in key:
                    continue
                original_key = key.replace("model_ema.", "")
                original_image_key = ema_mapping_temp[original_key]
                image_ema_mapping[key] = original_image_key
                original_ema_params[key] = value
        ##inflated encoder
        for (image_key, image_value), (video_key, video_value) in zip(original_image_encoder_params.items(), new_video_encoder_params.items()):
            if image_value.shape == video_value.shape: #not conv function
                new_video_encoder_params[video_key] = image_value
                continue
            if list(image_value.shape) == (list(video_value.shape[:2]) + list(video_value.shape[3:])): # conv function using centeral inflation [out_cha, in_chann, k_h, k_w]
                weight_3d = torch.zeros(*video_value.shape)
                # middle_idx = weight_3d.shape[2] // 2
                last_idx = weight_3d.shape[2] - 1 
                weight_3d[:, :, last_idx, :, :] = image_value
                new_video_encoder_params[video_key] = weight_3d
                continue
        self.encoder.load_state_dict(new_video_encoder_params, strict=True)
        
        ### inflated decoder
        for (image_key, image_value), (video_key, video_value) in zip(original_image_decoder_params.items(), new_video_decoder_params.items()):
            if image_value.shape == video_value.shape: #not conv function
                new_video_decoder_params[video_key] = image_value
                continue
            elif list(image_value.shape) == (list(video_value.shape[:2]) + list(video_value.shape[3:])): # conv function using centeral inflation [out_cha, in_chann, k_h, k_w]
                weight_3d = torch.zeros(*video_value.shape)
                # middle_idx = weight_3d.shape[2] // 2
                last_idx = weight_3d.shape[2] - 1
                weight_3d[:, :, last_idx, :, :] = image_value
                new_video_decoder_params[video_key] = weight_3d
                continue
            else: ## handle two with different channels
                if len(video_value.shape) > 1: #weight
                    weight_3d = torch.zeros(*video_value.shape)
                    last_idx = weight_3d.shape[2] - 1 ##temporally last slice 
                    weight_3d[:, :, last_idx, :, :] = image_value.repeat(2, 1, 1, 1)
                else: # bias
                    weight_3d = torch.zeros(*video_value.shape)
                    weight_3d = image_value.repeat(2)
                new_video_decoder_params[video_key] = weight_3d
        self.decoder.load_state_dict(new_video_decoder_params, strict=True)

        ## inflate EMA params
        for image_key, image_value in original_ema_params.items():
            original_image_key = image_ema_mapping[image_key]
            if "conv" in original_image_key or "downsample" in original_image_key or "nin_shortcut" in original_image_key:
                split_keys = original_image_key.split(".")
                split_keys.insert(-1, "conv_1")
                video_key = ".".join(split_keys)
            else:
                video_key = original_image_key
            ema_video_key = self.model_ema.m_name2s_name[video_key]
            video_value = getattr(self.model_ema, ema_video_key)
            if image_value.shape == video_value.shape:
                setattr(self.model_ema, ema_video_key, image_value)
            elif list(image_value.shape) == (list(video_value.shape[:2]) + list(video_value.shape[3:])):
                weight_3d = torch.zeros(*video_value.shape)
                # middle_idx = weight_3d.shape[2] // 2
                last_idx = weight_3d.shape[2] - 1
                weight_3d[:, :, last_idx, :, :] = image_value
                setattr(self.model_ema, ema_video_key, weight_3d)
            else: #handle decoder
                if len(video_value.shape) > 1: #weight
                    weight_3d = torch.zeros(*video_value.shape)
                    last_idx = weight_3d.shape[2] - 1 ##temporally last slice 
                    weight_3d[:, :, last_idx, :, :] = image_value.repeat(2, 1, 1, 1)
                else: # bias
                    weight_3d = torch.zeros(*video_value.shape)
                    weight_3d = image_value.repeat(2)
                setattr(self.model_ema, ema_video_key, weight_3d)
        
        print(f"Inflated from {image_pretrain_path}")

    def load_state_dict(self, *args, strict=False):
        """
        Resume not strict loading
        """
        return super().load_state_dict(*args, strict=strict)

    def state_dict(self, *args, destination=None, prefix='', keep_vars=False):
        return {k: v for k, v in super().state_dict(*args, destination, prefix, keep_vars).items() if ("inception_model" not in k and "lpips_vgg" not in k and "lpips_alex" not in k)}
        
    def init_from_ckpt(self, path, ignore_keys=list(), stage="transformer"):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        ema_mapping = {}
        new_params = OrderedDict()
        if stage == "transformer": ### directly use ema encoder and decoder parameter
            if self.use_ema:
                for k, v in sd.items(): 
                    if "encoder" in k:
                        if "model_ema" in k:
                            k = k.replace("model_ema.", "") #load EMA Encoder or Decoder
                            new_k = ema_mapping[k]
                            new_params[new_k] = v   
                        s_name = k.replace('.', '')
                        ema_mapping.update({s_name: k})
                        continue
                    if "decoder" in k:
                        if "model_ema" in k:
                            k = k.replace("model_ema.", "") #load EMA Encoder or Decoder
                            new_k = ema_mapping[k]
                            new_params[new_k] = v 
                        s_name = k.replace(".", "")
                        ema_mapping.update({s_name: k})
                        continue 
            else: #also only load the Generator
                for k, v in sd.items():
                    if "encoder" in k:
                        new_params[k] = v
                    elif "decoder" in k:
                        new_params[k] = v                  
        missing_keys, unexpected_keys = self.load_state_dict(new_params, strict=False) #first stage
        print(f"Restored from {path}")

    def encode(self, x):
        h = self.encoder(x)
        # h = self.quant_conv(h)
        (quant, emb_loss, info), loss_breakdown = self.quantize(h, return_loss_breakdown=True)
        return quant, emb_loss, info, loss_breakdown

    def decode(self, quant):
        # quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    def decode_code(self, code_b):
        quant_b = self.quantize.embed_code(code_b)
        dec = self.decode(quant_b)
        return dec

    def forward(self, input):
        quant, diff, _, loss_break = self.encode(input)
        dec = self.decode(quant)
        return dec, diff, loss_break

    def get_input(self, batch, k):
        x = batch[k]
        if k == "image":
            if len(x.shape) == 3:
                x = x[..., None]
            x = x.permute(0, 3, 1, 2).contiguous()
        return x
    
    def on_train_start(self):
        """
        change lr after resuming
        """
        if self.resume_lr is not None:
            opt_gen, opt_disc = self.optimizers()
            for opt_gen_param_group, opt_disc_param_group in zip(opt_gen.param_groups, opt_disc.param_groups):
                opt_gen_param_group["lr"] = self.resume_lr
                opt_disc_param_group["lr"] = self.resume_lr

    # fix mulitple optimizer bug
    # refer to https://lightning.ai/docs/pytorch/stable/model/manual_optimization.html
    def training_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        xrec, eloss,  loss_break = self(x)

        ###Adjuts the learning rate
        if self.sche_type is not None and self.resume_lr is None:
            g_it = self.trainer.global_step
            if self.max_it is None:
                iters_train = len(self.trainer.train_dataloader) ## get the total iterations in a epoch
                max_it = self.trainer.max_epochs * iters_train
                wp_it = self.wp * iters_train
            else:
                max_it = max_it
                wp_it = self.wp_iter
            self.lr_annealing(self.learning_rate, g_it, wp_it, max_it, wp0=self.wp0, wpe=self.wpe)

        opt_gen, opt_disc = self.optimizers()
        # scheduler_gen, scheduler_disc = self.lr_schedulers()

        ####################
        # fix global step bug
        # refer to https://github.com/Lightning-AI/pytorch-lightning/issues/17958
        opt_disc._on_before_step = lambda: self.trainer.profiler.start("optimizer_step")
        opt_disc._on_after_step = lambda: self.trainer.profiler.stop("optimizer_step")
        # opt_gen._on_before_step = lambda: self.trainer.profiler.start("optimizer_step")
        # opt_gen._on_after_step = lambda: self.trainer.profiler.stop("optimizer_step")
        ####################
        # original VQGAN first optimizes G, then D. We first optimize D then G, following traditional GAN
        
        # optimize generator
        aeloss, log_dict_ae = self.loss(eloss, loss_break, x, xrec, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="train")
        opt_gen.zero_grad()
        self.manual_backward(aeloss)
        opt_gen.step()
        
        # optimize discriminator
        discloss, log_dict_disc = self.loss(eloss, loss_break, x, xrec, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
        opt_disc.zero_grad()
        self.manual_backward(discloss)
        opt_disc.step()

        self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)


        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)

    def on_train_epoch_start(self):
        if self.use_shared_epoch: ## set the epoch for determinstic
            self.trainer.train_dataloader.dataset.epoch.set_value(self.trainer.current_epoch)

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self)
    
    def lr_annealing(self, peak_lr, cur_it, wp_it, max_it, wp0=0.005, wpe=0.001):
        """
        Modified from VAR
        """
        wp_it = round(wp_it)
        if cur_it < wp_it:
            cur_lr = wp0 + (1-wp0) * cur_it / wp_it
        else:
            pasd = (cur_it - wp_it) / (max_it-1 - wp_it)   # [0, 1]
            rest = 1 - pasd     # [1, 0]
            if self.sche_type == "lin0":
                T = 0.05; max_rest = 1-T
                if pasd < T: cur_lr = 1
                else: cur_lr = wpe + (1-wpe) * rest / max_rest
            if self.sche_type == "cos":
                cur_lr = wpe + (1-wpe) * (0.5 + 0.5 * math.cos(math.pi * pasd))

            if self.sche_type == "constant": ##warmup +  constant
                cur_lr = 1.0

        cur_lr *= peak_lr
        pasd = cur_it / (max_it-1)

        inf = 1e6
        min_lr, max_lr = inf, -1
        opt_gen, opt_disc = self.optimizers()
        ### adjust Generator Learning Rate
        for param_group in opt_gen.param_groups:
            param_group['lr'] = cur_lr * param_group.get('lr_sc', 1)    # 'lr_sc' could be assigned

        ### adjust Discriminator Learning Rate
        for param_group in opt_disc.param_groups:
            param_group['lr'] = cur_lr * param_group.get('lr_sc', 1)    # 'lr_sc' could be assigned

    def validation_step(self, batch, batch_idx): 
        if self.use_ema:
            with self.ema_scope():
                log_dict_ema = self._validation_step(batch, batch_idx, suffix="_ema")
        else:
            log_dict = self._validation_step(batch, batch_idx)

    def _validation_step(self, batch, batch_idx, suffix=""):
        x = self.get_input(batch, self.image_key)
        quant, eloss, indices, loss_break = self.encode(x)
        x_rec = self.decode(quant).clamp(-1, 1)
        aeloss, log_dict_ae = self.loss(eloss, loss_break, x, x_rec, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="val"+ suffix)

        discloss, log_dict_disc = self.loss(eloss, loss_break, x, x_rec, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val" + suffix)
    
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        
        return self.log_dict

    def test_step(self, batch, batch_idx):
        if self.use_ema:
            with self.ema_scope():
                log_dict_ema = self._test_step(batch, batch_idx, suffix="_ema")
        else:
            log_dict = self._test_step(batch, batch_idx)
        

    def _test_step(self, batch, batch_idx, suffix=""):
        x = self.get_input(batch, self.image_key)
        xrec, eloss,  loss_break = self(x)
        x_rec = xrec.clamp(-1, 1)
        aeloss, log_dict_ae = self.loss(eloss, loss_break, x, x_rec, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="val"+ suffix)

        discloss, log_dict_disc = self.loss(eloss, loss_break, x, x_rec, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val" + suffix)
    
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)

        return self.log_dict
    
    def configure_optimizers(self):
        lr = self.learning_rate
        opt_gen = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quantize.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))
        if self.trainer.is_global_zero:
            print("step_per_epoch: {}".format(len(self.trainer.datamodule._train_dataloader()) // self.trainer.world_size))
        step_per_epoch  = len(self.trainer.datamodule._train_dataloader()) // self.trainer.world_size
        warmup_steps = step_per_epoch * self.warmup_epochs
        training_steps = step_per_epoch * self.trainer.max_epochs

        if self.scheduler_type == "None":
            return ({"optimizer": opt_gen}, {"optimizer": opt_disc})
    
        if self.scheduler_type == "linear-warmup":
            scheduler_ae = torch.optim.lr_scheduler.LambdaLR(opt_gen, Scheduler_LinearWarmup(warmup_steps))
            scheduler_disc = torch.optim.lr_scheduler.LambdaLR(opt_disc, Scheduler_LinearWarmup(warmup_steps))

        elif self.scheduler_type == "linear-warmup_cosine-decay":
            multipler_min = self.min_learning_rate / self.learning_rate
            scheduler_ae = torch.optim.lr_scheduler.LambdaLR(opt_gen, Scheduler_LinearWarmup_CosineDecay(warmup_steps=warmup_steps, max_steps=training_steps, multipler_min=multipler_min))
            scheduler_disc = torch.optim.lr_scheduler.LambdaLR(opt_disc, Scheduler_LinearWarmup_CosineDecay(warmup_steps=warmup_steps, max_steps=training_steps, multipler_min=multipler_min))
        else:
            raise NotImplementedError()
        return {"optimizer": opt_gen, "lr_scheduler": scheduler_ae}, {"optimizer": opt_disc, "lr_scheduler": scheduler_disc}

    def get_last_layer(self):
        return self.decoder.conv_out.conv_1.weight

    def log_images(self, batch, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        xrec, _ = self(x)
        if x.shape[1] > 3:
            # colorize with random projection
            assert xrec.shape[1] > 3
            x = self.to_rgb(x)
            xrec = self.to_rgb(xrec)
        log["inputs"] = x
        log["reconstructions"] = xrec
        return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
        return x