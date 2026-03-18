"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import logging
import math

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
from torch.nn import functional as F
from lavis.common.registry import registry
from lavis.models.base_model import all_gather_with_grad, concat_all_gather
from lavis.models.blip2_models.blip2 import (
    Blip2Base,
    compute_sim_matrix,
    disabled_train,
)
from lavis.models.blip_models.blip_outputs import BlipOutput, BlipOutputFeatures
from sklearn.cluster import KMeans
from sklearn.metrics import mutual_info_score
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.cluster import DBSCAN


def l2norm(X, dim=-1):
    """L2-normalize columns of X"""
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt()
    X = torch.div(X, norm)
    return X


def l1norm(X, dim):
    """L1-normalize columns of X"""
    norm = torch.abs(X).sum(dim=dim, keepdim=True)
    X = torch.div(X, norm)
    return X


def info_nce(query, target):
    bs = query.size(0)
    targets = torch.linspace(0, bs - 1, bs, dtype=int).to(query.device)
    temp = nn.Parameter(0.07 * torch.ones([]))
    x = torch.matmul(query, target).squeeze().to(query.device)
    # print('x',x.shape)
    sim_i2t, _ = x.max(-1)
    sim_i2t = sim_i2t / temp
    return F.cross_entropy(sim_i2t, targets)


class SoftEstimationMarginLoss(nn.Module):

    def __init__(self, margin=1):
        super(SoftEstimationMarginLoss, self).__init__()
        self.margin = margin

    def forward(
            self,
            fusion_feats,  # 组合特征
            target_feats,  # 目标特征
            hard_negative=True,
            labels=None,
            soft_margin="exponential",
            mode="train",
    ):
        
        fusion_feats_ = fusion_feats.unsqueeze(1).unsqueeze(1)
        target_feats_ = target_feats.permute(0, 2, 1)
        x = torch.matmul(fusion_feats_, target_feats_).squeeze().to(fusion_feats_.device)
        scores, _ = x.max(-1)
        bs = x.shape[0]

        
        diagonal = scores.diag().view(scores.size(0), 1)
        d1 = diagonal.expand_as(scores)

        
        if labels is None:
            margin = self.margin
        elif soft_margin == "linear":
            margin = self.margin * labels.view(-1, 1)  
        elif soft_margin == "exponential":
            s = (torch.pow(10, labels) - 1) / 9
            margin = self.margin * s.view(-1, 1)
        elif soft_margin == "sin":
            s = torch.sin(math.pi * labels - math.pi / 2) / 2 + 1 / 2
            margin = self.margin * s.view(-1, 1)

        
        cost_s = (margin + scores - d1).clamp(min=0)
        
        mask = torch.eye(scores.size(0)) > 0.5
        mask = mask.to(cost_s.device)
        cost_s = cost_s.masked_fill_(mask, 0)

        
        cost_s_max = cost_s.max(1)[0]

        return cost_s_max.sum() / bs 


@registry.register_model("HABIT")
class HABIT(Blip2Base):
    """
    BLIP2 first-stage model with Q-former and ViT.
    Supported model types:
        - pretrained: pretrained model with vit-g
        - pretrain_vitL: pretrained model with vit-large
        - coco: fintuned model on coco
    Usage:
        >>> from lavis.models import load_model
        >>> model = load_model("blip2", "pretrain")
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain": "configs/models/blip2/blip2_pretrain.yaml",
        "pretrain_vitL": "configs/models/blip2/blip2_pretrain_vitL.yaml",
        "coco": "configs/models/blip2/blip2_coco.yaml",
    }

    def __init__(
            self,
            vit_model="eva_clip_g",
            img_size=224,
            drop_path_rate=0,
            use_grad_checkpoint=False,
            vit_precision="fp16",
            freeze_vit=True,
            num_query_token=32,
            cross_attention_freq=2,
            embed_dim=256,
            max_txt_len=32,
    ):
        super().__init__()

        self.tokenizer = self.init_tokenizer()
        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision
        )
        if freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            logging.info("freeze vision encoder")
        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, self.visual_encoder.num_features, cross_attention_freq
        )
        self.Qformer.resize_token_embeddings(len(self.tokenizer))
        state_dict = self.Qformer.state_dict()
        for name, param in self.Qformer.named_parameters():
            if "_query" in name:
                key_orig = name.replace("_query", "")
                param.data.copy_(state_dict[key_orig])

        self.vision_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)
        self.text_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)

        self.itm_head = nn.Linear(self.Qformer.config.hidden_size, 2)

        self.temp = nn.Parameter(0.07 * torch.ones([]))

        self.max_txt_len = max_txt_len
        self.prompt_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, self.Qformer.config.hidden_size)
        )
        self.prompt_tokens.data.normal_(mean=0.0, std=self.Qformer.config.initializer_range)
        self.soft_margin_loss = SoftEstimationMarginLoss(margin=0.5)
        self.soft_labels = []
        self.sim = []
        self.KLDivLoss = nn.KLDivLoss(reduction='mean')

    def robust_infoNCE(self, query, target, mask_sample):
        
        eps = 1e-7
        bs = query.size(0)
        x = torch.matmul(query, target).squeeze().to(query.device)
        sim_i2t, _ = x.max(-1)
        i2t = (sim_i2t / 0.07).softmax(1)  # 适用于批次样本输入计算损失
        i2t = torch.clamp(i2t, min=eps, max=1 - eps)
        mask_sample = torch.tensor(mask_sample).to(query.device)
        mask_sample = mask_sample.unsqueeze(1).expand(-1, 256)
        i2t = i2t * mask_sample
        labels = torch.arange(query.shape[0]).long().cuda()
        mask = torch.ones_like(i2t).to(float).to(i2t.device)
        mask[torch.arange(bs), labels] = 0.
        loss = - ((1. - i2t).log() * mask).sum() / bs
        row_losses = - ((1. - i2t).log() * mask).sum(dim=1)

        min_loss_idx = row_losses.argmin()
        return loss, min_loss_idx, sim_i2t
    
    def cal_Knowledge_Consistency(self,sim_Historical,sim_Current,Historical_soft_labels,Current_soft_labels):
        Historical_soft_labels = Historical_soft_labels.unsqueeze(1).expand(-1, 256)
        Historical_soft_labels = Historical_soft_labels.detach()
        Current_soft_labels = Current_soft_labels.unsqueeze(1).expand(-1, 256)
        sim_Historical_masked = sim_Historical * Historical_soft_labels*10
        sim_Current_masked= sim_Current * Current_soft_labels*10
        
        sim_Current_prob = torch.softmax(sim_Current_masked, dim=1) 
        sim_Current_logprob = torch.log(sim_Current_prob) 

        
        sim_Historical_prob = torch.softmax(sim_Historical_masked, dim=1) 
        
        divergence = self.KLDivLoss(sim_Current_logprob, sim_Historical_prob)
        return divergence



    def estimate_mutual_knowledge(self, x, y, bins=10):
        
        if x.shape != y.shape:
            raise ValueError(f"输入特征维度必须匹配。Got {x.shape} 和 {y.shape}")

        
        x = x.detach().cpu().numpy().flatten()  
        y = y.detach().cpu().numpy().flatten() 

        d = x.shape[0]  
        
        x_bins = np.linspace(x.min(), x.max(), bins + 1)  
        x_counts, _ = np.histogram(x, bins=x_bins)
        p_x = x_counts / d  

        
        y_bins = np.linspace(y.min(), y.max(), bins + 1)
        y_counts, _ = np.histogram(y, bins=y_bins)
        p_y = y_counts / d

        
        joint_counts, _, _ = np.histogram2d(x, y, bins=[x_bins, y_bins])
        p_xy = joint_counts / d  
        
        epsilon = 1e-10
        p_x = np.expand_dims(p_x, axis=1)  # 形状 [bins, 1]
        p_y = np.expand_dims(p_y, axis=0)  # 形状 [1, bins]
        denominator = p_x * p_y  # 形状 [bins, bins]，p(x)p(y)
        mi = np.sum(p_xy * np.log((p_xy + epsilon) / (denominator + epsilon)))

        return mi

    
    def calculate_mk_transition_rate(self, anchor_img, anchor_text, img, text):
        
        mi_anchor = self.estimate_mutual_knowledge(anchor_img, anchor_text)
        mi_current = self.estimate_mutual_knowledge(img, text)
        mi_text = self.estimate_mutual_knowledge(anchor_img, text)
        mi_image = self.estimate_mutual_knowledge(img, anchor_text)
        
        transition_rate = abs(mi_anchor - mi_current) / mi_anchor
        transition_rate_for_text = abs(mi_anchor - mi_text) / mi_anchor
        transition_rate_for_image = abs(mi_anchor - mi_image) / mi_anchor
        return transition_rate, transition_rate_for_text, transition_rate_for_image

    
    def calculate_soft_label(self, transition_rate, transition_rate_for_text, transition_rate_for_image):
        return 1 / (1 + abs(transition_rate) + abs(transition_rate_for_text - transition_rate_for_image))

    def forward(self, samples, device,Historical_soft_labels_samples,epoch,similarity_matrix_Historical_samples):
        image = samples["image"]
        target = samples["target"]
        text = samples["text_input"]

        
        image_embeds = self.ln_vision(self.visual_encoder(image))
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )
        
        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
            self.device
        )
        
        text_tokens = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(image.device)
        
        attention_mask = torch.cat([query_atts, text_tokens.attention_mask], dim=1)
        fusion_output = self.Qformer.bert(
            text_tokens.input_ids,
            query_embeds=query_tokens,
            attention_mask=attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )

        taregt_embeds = self.ln_vision(self.visual_encoder(target))
        target_atts = torch.ones(taregt_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )
        target_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=taregt_embeds,
            encoder_attention_mask=target_atts,
            use_cache=True,
            return_dict=True,
        )
        
        target_feats = F.normalize(
            self.vision_proj(target_output.last_hidden_state), dim=-1
        )
        target_feats_for_labels = target_feats

        
        fusion_feats = F.normalize(
            self.text_proj(fusion_output.last_hidden_state[:, 32, :]), dim=-1
        )
        fusion_feats_for_labels = F.normalize(
            self.text_proj(fusion_output.last_hidden_state[:, :32, :]), dim=-1
        )

        fusion_feats_ = fusion_feats.unsqueeze(1).unsqueeze(1)
        target_feats_ = target_feats.permute(0, 2, 1)
        mask_sample = torch.ones(256,device = image.device)
        _, min_loss_idx,_ = self.robust_infoNCE(fusion_feats_.detach(), target_feats_.detach(), mask_sample)

        
        batch_size = image.size(0)
        soft_labels = torch.zeros(batch_size, device=device)
        
        for i in range(batch_size):
            sample_fusion_feat = fusion_feats_for_labels[i:i + 1].detach()
            sample_target_feat = target_feats_for_labels[i:i + 1].detach()
            
            transition_rate, transition_rate_for_text, transition_rate_for_image = self.calculate_mk_transition_rate(
                fusion_feats_for_labels[min_loss_idx:min_loss_idx + 1],
                target_feats_for_labels[min_loss_idx:min_loss_idx + 1], sample_fusion_feat, sample_target_feat)
            soft_labels[i] = self.calculate_soft_label(transition_rate, transition_rate_for_text, transition_rate_for_image)

        if (epoch >= 1):
            dbscan_Historical = DBSCAN(eps=0.01, min_samples=10, n_jobs=-1)
            labels_Historical = dbscan_Historical.fit_predict(Historical_soft_labels_samples.cpu().numpy().reshape(-1, 1))
            dbscan_Current = DBSCAN(eps=0.01, min_samples=10, n_jobs=-1)
            labels_Current = dbscan_Current.fit_predict(soft_labels.cpu().numpy().reshape(-1, 1))
            
            mask = np.ones(batch_size)  
            
            for i in range(batch_size):
                if labels_Historical[i] == -1 and labels_Current[i] == -1:
                    mask[i] = 0  
        else:
            mask = np.ones(batch_size)

        
        hard_negative = True
        self.soft_labels.append(soft_labels.view(-1))
        mask_sample = torch.tensor(mask).to(image.device)
        
        loss_rank, min_loss_idx,sim = self.robust_infoNCE(fusion_feats_, target_feats_, mask_sample)
        self.sim.append(sim.detach())
        soft_labels = soft_labels * mask_sample
        loss_soft = self.soft_margin_loss(fusion_feats, target_feats, hard_negative=hard_negative, labels=soft_labels)
        if epoch >= 1:
            kl_loss = self.cal_Knowledge_Consistency(sim, similarity_matrix_Historical_samples, Historical_soft_labels_samples, soft_labels)
        else:
            kl_loss = 0
        return {'loss_rank': loss_rank, 'loss_soft': loss_soft,'loss_kl': kl_loss}


    @torch.no_grad()
    def extract_retrieval_compose(self, img, mod, return_attns=False):
        with self.maybe_autocast():
            image_embeds_frozen = self.ln_vision(self.visual_encoder(img))
        image_embeds_frozen = image_embeds_frozen.float()

        # return image_embeds
        reference_embeds = image_embeds_frozen

        image_atts = torch.ones(reference_embeds.size()[:-1], dtype=torch.long).to(
            reference_embeds.device
        )
        # query tokens
        query_tokens = self.query_tokens.expand(reference_embeds.shape[0], -1, -1)
        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
            self.device
        )
        # text tokens
        text_tokens = self.tokenizer(
            mod,
            padding="max_length",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(reference_embeds.device)

        attention_mask = torch.cat([query_atts, text_tokens.attention_mask], dim=1)
        fusion_output = self.Qformer.bert(
            text_tokens.input_ids,
            query_embeds=query_tokens,
            attention_mask=attention_mask,
            encoder_hidden_states=reference_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
            output_attentions=return_attns
        )


        fusion_feats = F.normalize(
            self.text_proj(fusion_output.last_hidden_state[:, 32, :]), dim=-1
        )

        return fusion_feats.unsqueeze(1).unsqueeze(1)

    @torch.no_grad()
    def extract_retrieval_target(self, img):
        with self.maybe_autocast():
            image_embeds_frozen = self.ln_vision(self.visual_encoder(img))
        image_embeds_frozen = image_embeds_frozen.float()
        image_atts = torch.ones(
            image_embeds_frozen.size()[:-1], dtype=torch.long
        ).to(self.device)
        query_tokens = self.query_tokens.expand(
            image_embeds_frozen.shape[0], -1, -1
        )

        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds_frozen,
            encoder_attention_mask=image_atts,
            return_dict=True,
            output_attentions=True
        )
        image_embeds = query_output.last_hidden_state
        image_features = F.normalize(self.vision_proj(image_embeds), dim=-1)
        return image_features.permute(0, 2, 1)

    @torch.no_grad()
    def extract_target_features(self, image, mode='mean'):
        with self.maybe_autocast():
            image_embeds_frozen = self.ln_vision(self.visual_encoder(image))
        image_embeds_frozen = image_embeds_frozen.float()
        image_atts = torch.ones(
            image_embeds_frozen.size()[:-1], dtype=torch.long
        ).to(self.device)
        query_tokens = self.query_tokens.expand(
            image_embeds_frozen.shape[0], -1, -1
        )

        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds_frozen,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        image_embeds = query_output.last_hidden_state

        # return image_embeds
        image_features = F.normalize(self.vision_proj(image_embeds), dim=-1)
        return image_features, image_embeds_frozen

    @torch.no_grad()
    def extract_features(self, samples, mode="multimodal"):
        """
        Extract features for multimodal or unimodal samples.
        Args:
            samples (dict): A dictionary of samples, containing the following keys:
                - image (torch.Tensor): A tensor of shape (B, C, H, W) containing the image.
                    Raw images should be preprocessed before being passed to feature extractor.
                - text_input (list): A list of strings containing the text, length B.
            mode (str): The mode of feature extraction. Can be either "multimodal", "text" or "image".
                If "multimodal", return image features and multimodal features;
                if "text", return text features;
                if "image", return image features.
                Default: "multimodal".
        Returns:
            BlipOutputFeatures: A BlipOutputFeatures object containing the features.
                See lavis/models/blip_models/blip_outputs.py for more details.
        """
        image = samples.get("image")
        caption = samples.get("text_input")

        # assert mode is one of "image", "text", "multimodal"
        assert mode in [
            "image",
            "text",
            "multimodal",
        ], "mode must be one of 'image', 'text', 'multimodal'"

        # initalize output
        image_embeds, text_embeds, multimodal_embeds = None, None, None
        image_features, text_features = None, None

        if mode == "image":
            assert (
                    image is not None
            ), "Image is not provided for mode 'image' or 'multimodal'"
            # return query features
            with self.maybe_autocast():
                image_embeds_frozen = self.ln_vision(self.visual_encoder(image))
            image_embeds_frozen = image_embeds_frozen.float()
            image_atts = torch.ones(
                image_embeds_frozen.size()[:-1], dtype=torch.long
            ).to(self.device)
            query_tokens = self.query_tokens.expand(
                image_embeds_frozen.shape[0], -1, -1
            )

            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds_frozen,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
            image_embeds = query_output.last_hidden_state
            image_features = F.normalize(self.vision_proj(image_embeds), dim=-1)

        elif mode == "text":
            assert (
                    caption is not None
            ), "text input is None for mode 'text' or 'multimodal'"

            # return text features
            text = self.tokenizer(caption, return_tensors="pt", padding=True).to(
                self.device
            )

            text_output = self.Qformer.bert(
                text.input_ids,
                attention_mask=text.attention_mask,
                return_dict=True,
            )
            text_embeds = text_output.last_hidden_state
            text_features = self.text_proj(text_embeds)
            text_features = F.normalize(text_features, dim=-1)

        elif mode == "multimodal":
            # return multimodel query features
            with self.maybe_autocast():
                image_embeds_frozen = self.ln_vision(self.visual_encoder(image))
            image_embeds_frozen = image_embeds_frozen.float()
            image_atts = torch.ones(
                image_embeds_frozen.size()[:-1], dtype=torch.long
            ).to(self.device)
            query_tokens = self.query_tokens.expand(
                image_embeds_frozen.shape[0], -1, -1
            )
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
                self.device
            )

            text = self.tokenizer(caption, return_tensors="pt", padding=True).to(
                self.device
            )
            attention_mask = torch.cat([query_atts, text.attention_mask], dim=1)

            output = self.Qformer.bert(
                text.input_ids,
                query_embeds=query_tokens,
                attention_mask=attention_mask,
                encoder_hidden_states=image_embeds_frozen,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

            multimodal_embeds = output.last_hidden_state[:, : query_tokens.size(1), :]

        return BlipOutputFeatures(
            image_embeds=image_embeds,
            image_embeds_proj=image_features,
            text_embeds=text_embeds,
            text_embeds_proj=text_features,
            multimodal_embeds=multimodal_embeds,
        )

    @classmethod
    def from_config(cls, cfg):
        vit_model = cfg.get("vit_model", "eva_clip_g")
        img_size = cfg.get("image_size")
        num_query_token = cfg.get("num_query_token")
        cross_attention_freq = cfg.get("cross_attention_freq", 2)

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)

        max_txt_len = cfg.get("max_txt_len", 32)

        model = cls(
            vit_model=vit_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            num_query_token=num_query_token,
            cross_attention_freq=cross_attention_freq,
            max_txt_len=max_txt_len,
        )
        model.load_checkpoint_from_config(cfg)

        return model

    def compute_sim_matrix(self, data_loader, task_cfg):
        """
        Compute similarity i2t, t2i matrix for the given data loader.
        """
        k_test = task_cfg.k_test

        return compute_sim_matrix(model=self, data_loader=data_loader, k_test=k_test)

    def get_soft_labels(self):
        flatten_soft_labels = torch.cat(self.soft_labels)
        # print(self.soft_labels)
        self.soft_labels = []
        return flatten_soft_labels

    def get_similarity_matrix(self):
        return self.sim