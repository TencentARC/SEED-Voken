import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import lightning as L
import math
import os
import sys
sys.path.append(os.getcwd())
from main import instantiate_from_config

from IBQ.modules.diffusionmodules.model import Encoder, Decoder
from IBQ.modules.vqvae.quantize import VectorQuantizer2 as VectorQuantizer
from IBQ.modules.vqvae.quantize import IndexPropagationQuantize
from IBQ.modules.scheduler.lr_scheduler import Scheduler_LinearWarmup, Scheduler_LinearWarmup_CosineDecay
from IBQ.modules.util import requires_grad
from IBQ.modules.ema import LitEma
from collections import OrderedDict
from contextlib import contextmanager

class VQModel(L.LightningModule):
    def __init__(self,
                ddconfig,
                lossconfig,
                n_embed,
                embed_dim,
                ckpt_path=None,
                ignore_keys=[],
                image_key="image",
                colorize_nlabels=None,
                monitor=None,
                remap=None,
                sane_index_shape=False,  # tell vector quantizer to return indices as bhw
                learning_rate=None,
                l2_normalize=False,
                ### scheduler config
                warmup_epochs=0,  # warmup epochs
                scheduler_type="None",
                min_learning_rate=0,
                gradient_clip_val=0,
                resume_lr=None,
                use_ema=False,
                stage=None,
                wp = 0,
                wp0 = 0.005, #initial lr ratio at the begging of lr warm up
                wpe = 0.01, #final lr ratio at the end of training
                max_iter = None,
                sche_type = None,
                wp_iter = None,
                use_shared_epoch = True,
                lr_drop_iter = None,
                lr_drop_rate = 0.1,
                ):
        super().__init__()
        self.image_key = image_key
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.loss = instantiate_from_config(lossconfig)
        self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25,
                                        remap=remap, sane_index_shape=sane_index_shape, l2_normalize=l2_normalize)
        self.quant_conv = torch.nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        self.use_ema = use_ema
        self.stage = stage
        self.image_key = image_key
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor
        
        if self.use_ema and stage is None:  # no need to construct ema when training transformer
            self.model_ema = LitEma(self)
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys, stage=stage)

        self.learning_rate = learning_rate
        self.automatic_optimization = False
        self.scheduler_type = scheduler_type
        self.warmup_epochs = warmup_epochs
        self.min_learning_rate = min_learning_rate
        self.gradient_clip_val = gradient_clip_val
        self.resume_lr = resume_lr

        ## for scheduler
        self.wp = wp
        self.wp0 = wp0
        self.wpe = wpe
        self.sche_type = sche_type
        self.max_it = max_iter
        self.wp_iter = wp_iter
        self.use_shared_epoch = use_shared_epoch
        self.lr_drop_iter = lr_drop_iter
        self.lr_drop_rate = lr_drop_rate

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

    def load_state_dict(self, *args, strict=False):
        """
        Resume not strict loading
        """
        return super().load_state_dict(*args, strict=strict)

    def state_dict(self, *args, destination=None, prefix='', keep_vars=False):
        return {k: v for k, v in super().state_dict(*args, destination, prefix, keep_vars).items() if ("inception_model" not in k and "lpips_vgg" not in k and "lpips_alex" not in k)}

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

        cur_lr *= peak_lr
        pasd = cur_it / (max_it-1)

        inf = 1e6
        min_lr, max_lr = inf, -1
        opt_gen, opt_disc = self.optimizers()
        ### adjust Generator Learning Rate
        for param_group in opt_gen.param_groups:
            param_group['lr'] = cur_lr * param_group.get('lr_sc', 1)    # 'lr_sc' could be assigned
            max_lr = max(max_lr, param_group['lr'])
            min_lr = min(min_lr, param_group['lr'])
        
        ### adjust Discriminator Learning Rate
        for param_group in opt_disc.param_groups:
            param_group['lr'] = cur_lr * param_group.get('lr_sc', 1)    # 'lr_sc' could be assigned
            max_lr = max(max_lr, param_group['lr'])
            min_lr = min(min_lr, param_group['lr'])

        if min_lr == inf: min_lr = -1
        return min_lr, max_lr

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
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    def decode_code(self, code_b):
        quant_b = self.quantize.embed_code(code_b)
        dec = self.decode(quant_b)
        return dec

    def forward(self, input):
        quant, diff, _ = self.encode(input)
        dec = self.decode(quant)
        return dec, diff

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        # x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x.float()

    def on_train_start(self):
        """
        change lr after resuming
        """
        if self.resume_lr is not None:
            opt_gen, opt_disc = self.optimizers()
            for opt_gen_param_group, opt_disc_param_group in zip(opt_gen.param_groups, opt_disc.param_groups):
                opt_gen_param_group["lr"] = self.resume_lr
                opt_disc_param_group["lr"] = self.resume_lr

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self)

    # fix mulitple optimizer bug
    # refer to https://lightning.ai/docs/pytorch/stable/model/manual_optimization.html
    def training_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        xrec, qloss = self(x)

        ###Adjusts the learning rate
        if self.sche_type is not None and self.resume_lr is None:
            g_it = self.trainer.global_step
            if self.max_it is None:
                iters_train = len(self.trainer.train_dataloader) ## get the total iterations in a epoch
                max_it = self.trainer.max_epochs * iters_train
                wp_it = self.wp * iters_train
            else:
                max_it = self.max_it
                wp_it = self.wp_iter
            self.lr_annealing(self.learning_rate, g_it, wp_it, max_it, wp0=self.wp0, wpe=self.wpe)

        opt_gen, opt_disc = self.optimizers()
        if self.scheduler_type != "None":
            scheduler_gen, scheduler_disc = self.lr_schedulers()

        ####################
        # fix global step bug
        # refer to https://github.com/Lightning-AI/pytorch-lightning/issues/17958
        opt_disc._on_before_step = lambda: self.trainer.profiler.start("optimizer_step")
        opt_disc._on_after_step = lambda: self.trainer.profiler.stop("optimizer_step")
        ####################

        # original VQGAN first optimizes G, then D. We first optimize D then G, following traditional GAN

        # optimize discriminator
        discloss, log_dict_disc = self.loss(qloss, x, xrec, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
        opt_disc.zero_grad()
        self.manual_backward(discloss)
        opt_disc.step()
        self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)

        # optimize generator
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="train")
        opt_gen.zero_grad()
        self.manual_backward(aeloss)


        if self.gradient_clip_val > 0: # for cosine similarity
            self.clip_gradients(opt_gen, gradient_clip_val=self.gradient_clip_val, gradient_clip_algorithm="norm")

        opt_gen.step()
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)

        if self.lr_drop_iter is not None:
            g_it = self.trainer.global_step
            if (g_it + 1) in self.lr_drop_iter:
                opt_gen, opt_disc = self.optimizers()
                for opt_gen_param_group, opt_disc_param_group in zip(opt_gen.param_groups, opt_disc.param_groups):
                    opt_gen_param_group["lr"] = opt_gen_param_group["lr"] * self.lr_drop_rate
                    opt_disc_param_group["lr"] = opt_disc_param_group["lr"] * self.lr_drop_rate

    def validation_step(self, batch, batch_idx):
        if self.use_ema:
            with self.ema_scope():
                log_dict_ema = self._validation_step(batch, batch_idx, suffix="_ema")
            return log_dict_ema
        else:
            log_dict = self._validation_step(batch, batch_idx)
            return log_dict

    def _validation_step(self, batch, batch_idx, suffix=""):
        x = self.get_input(batch, self.image_key)
        quant, qloss, (_, _, min_encoding_indices) = self.encode(x)
        x_rec = self.decode(quant).clamp(-1, 1)
        aeloss, log_dict_ae = self.loss(qloss, x, x_rec, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="val")

        discloss, log_dict_disc = self.loss(qloss, x, x_rec, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)

    def test_step(self, batch, batch_idx):
        if self.use_ema:
            with self.ema_scope():
                log_dict_ema = self._test_step(batch, batch_idx, suffix="_ema")
            return log_dict_ema
        else:
            log_dict = self._test_step(batch, batch_idx)
            return log_dict

    def _test_step(self, batch, batch_idx, suffix=""):
        x = self.get_input(batch, self.image_key)
        quant, qloss, (_, _, min_encoding_indices) = self.encode(x)
        x_rec = self.decode(quant).clamp(-1, 1)
        aeloss, log_dict_ae = self.loss(qloss, x, x_rec, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="val")

        discloss, log_dict_disc = self.loss(qloss, x, x_rec, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        
    def configure_optimizers(self):
        lr = self.learning_rate
        opt_gen = torch.optim.Adam(list(self.encoder.parameters()) +
                                   list(self.decoder.parameters()) +
                                   list(self.quantize.parameters()) +
                                   list(self.quant_conv.parameters()) +
                                   list(self.post_quant_conv.parameters()),
                                   lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))

        if self.scheduler_type == "None":
            return (
                {"optimizer": opt_gen},
                {"optimizer": opt_disc, "do_not_count_global_step": True},
            )

        if self.trainer.is_global_zero:
            print("step_per_epoch: {}".format(len(self.trainer.datamodule._train_dataloader()) // self.trainer.world_size))
            # import pdb; pdb.set_trace()
        step_per_epoch  = len(self.trainer.datamodule._train_dataloader()) // self.trainer.world_size
        warmup_steps = step_per_epoch * self.warmup_epochs
        training_steps = step_per_epoch * self.trainer.max_epochs

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
        return self.decoder.conv_out.weight

class IBQ(VQModel):
    def __init__(self,
                ddconfig,
                lossconfig,
                n_embed,
                embed_dim,
                ckpt_path=None,
                ignore_keys=[],
                image_key="image",
                colorize_nlabels=None,
                monitor=None,
                remap=None,
                sane_index_shape=False,  # tell vector quantizer to return indices as bhw
                learning_rate=None,
                l2_normalize=False,
                ### scheduler config
                warmup_epochs=0,  # warmup epochs
                scheduler_type="None",
                min_learning_rate=0,
                cosine_similarity=False,
                gradient_clip_val=0,
                use_entropy_loss=False,
                sample_minimization_weight=1.0,
                batch_maximization_weight=1.0,
                entropy_temperature=0.01,
                beta=0.25,
                resume_lr=None,
                use_ema=False,
                stage=None,
                wp = 0,
                wp0 = 0.005, #initial lr ratio at the begging of lr warm up
                wpe = 0.01, #final lr ratio at the end of training
                max_iter = None,
                sche_type = None,
                wp_iter = None,
                use_shared_epoch = True,
                lr_drop_iter = None,
                lr_drop_rate = 0.1,
                ):
        z_channels = ddconfig["z_channels"]
        super().__init__(ddconfig,
                        lossconfig,
                        n_embed,
                        embed_dim,
                        ckpt_path=None,
                        ignore_keys=ignore_keys,
                        image_key=image_key,
                        colorize_nlabels=colorize_nlabels,
                        monitor=monitor,
                        remap=remap,
                        sane_index_shape=sane_index_shape,
                        learning_rate=learning_rate,
                        l2_normalize=l2_normalize,
                        warmup_epochs=warmup_epochs,
                        scheduler_type=scheduler_type,
                        min_learning_rate=min_learning_rate,
                        gradient_clip_val=gradient_clip_val,
                        resume_lr=resume_lr,
                        use_ema=use_ema,
                        stage=stage,
                        wp = wp,
                        wp0 = wp0, #initial lr ratio at the begging of lr warm up
                        wpe = wpe, #final lr ratio at the end of training
                        max_iter = max_iter,
                        sche_type = sche_type,
                        wp_iter = wp_iter,
                        use_shared_epoch = use_shared_epoch,
                        lr_drop_iter = lr_drop_iter,
                        lr_drop_rate = lr_drop_rate
                        )
        self.quantize = IndexPropagationQuantize(n_embed, embed_dim, beta, use_entropy_loss,
                                          remap=remap, cosine_similarity=cosine_similarity,
                                          entropy_temperature=entropy_temperature,
                                          sample_minimization_weight=sample_minimization_weight, batch_maximization_weight=batch_maximization_weight)
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys, stage=stage)