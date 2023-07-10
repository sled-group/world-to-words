# The following file is originally adopted from the original MDETR codebase
# Copyright (c) Aishwarya Kamath & Nicolas Carion. Licensed under the Apache License 2.0. All Rights Reserved
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from typing import Optional
from transformers.models.roberta.modeling_roberta import RobertaLMHead
from transformers.models.roberta.configuration_roberta import RobertaConfig

import torch
import torch.distributed
import torch.nn.functional as F
from torch import nn

from einops import rearrange
from utils.misc import NestedTensor

from .backbone import build_backbone
from .matcher import build_matcher
from .segmentation import DETRsegm
from .transformer import build_transformer, ModelStage
from .criterion import MaskVisionContrastiveCriterion, QACriterionClevr, SetCriterion, MaskLanguageCriterion



class MDETR(nn.Module):
    """ This is the MDETR module that performs modulated object detection """

    def __init__(
        self,
        backbone,
        transformer,
        num_classes,
        num_queries,
        aux_loss=False,
        contrastive_hdim=64,
        contrastive_loss=False,
        contrastive_align_loss=False,
        mlm_loss = False,
        mvm_loss = False,
        mvm_prob = 0.4,
        qa_dataset: Optional[str] = None,
        split_qa_heads=True,
        predict_final=False,
        freeze_vision_encoder=False,
    ):
        """Initializes the model.

        Args:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         MDETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            contrastive_hdim: dimension used for projecting the embeddings before computing contrastive loss
            contrastive_loss: If true, perform image-text contrastive learning
            contrastive_align_loss: If true, perform box - token contrastive learning
            qa_dataset: If not None, train a QA head for the target dataset (CLEVR or GQA)
            split_qa_heads: If true, use several head for each question type
            predict_final: If true, will predict if a given box is in the actual referred set.
                           Useful for CLEVR-Ref+ only currently.
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.isfinal_embed = nn.Linear(hidden_dim, 1) if predict_final else None
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        if qa_dataset is not None:
            nb_heads = 6 if qa_dataset == "gqa" else 4
            self.qa_embed = nn.Embedding(nb_heads if split_qa_heads else 1, hidden_dim)

        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone
        if freeze_vision_encoder:
            for p in self.backbone.parameters():
                p.requires_grad_(False)
        self.aux_loss = aux_loss
        self.mlm_loss = mlm_loss
        self.mvm_loss = mvm_loss
        self.mvm_prob = mvm_prob
        # NOTE: Hard-coded vocab size as bert-base-uncased
        temp_mlm_config = RobertaConfig(hidden_size=hidden_dim, vocab_size=self.transformer.tokenizer.vocab_size, layer_norm_eps=1e-05)
        self.mlm_head = RobertaLMHead(temp_mlm_config)
        # TODO:
        self.mvm_head = torch.nn.Sequential(nn.Linear(hidden_dim, hidden_dim))
        self.learnt_vision_mask_feat = nn.parameter.Parameter(torch.ones(hidden_dim))
        self.contrastive_align_loss = contrastive_align_loss
        if contrastive_align_loss:
            # if do contrastive alignment, project the embedding to a lower, same dimension
            self.contrastive_align_projection_image = nn.Linear(hidden_dim, contrastive_hdim)
            self.contrastive_align_projection_text = nn.Linear(hidden_dim, contrastive_hdim)

        self.qa_dataset = qa_dataset
        self.split_qa_heads = split_qa_heads
        if qa_dataset is not None:
            if split_qa_heads:
                self.answer_type_head = nn.Linear(hidden_dim, 5)
                # TODO: make this more general
                if qa_dataset == "gqa":
                    self.answer_rel_head = nn.Linear(hidden_dim, 1594)
                    self.answer_obj_head = nn.Linear(hidden_dim, 3)
                    self.answer_global_head = nn.Linear(hidden_dim, 111)
                    self.answer_attr_head = nn.Linear(hidden_dim, 403)
                    self.answer_cat_head = nn.Linear(hidden_dim, 678)
                elif qa_dataset == "clevr":
                    self.answer_type_head = nn.Linear(hidden_dim, 3)
                    self.answer_binary_head = nn.Linear(hidden_dim, 1)
                    self.answer_attr_head = nn.Linear(hidden_dim, 15)
                    self.answer_reg_head = MLP(hidden_dim, hidden_dim, 20, 3)
                else:
                    assert False, f"Invalid qa dataset {qa_dataset}"
            else:
                # TODO: make this more general
                assert qa_dataset == "gqa", "Clevr QA is not supported with unified head"
                self.answer_head = nn.Linear(hidden_dim, 1853)


    def forward(self, samples: NestedTensor, captions, encode_and_save=True, memory_cache=None, lang_only=False, positive_map=None, one_pass=False):
        """The forward expects a NestedTensor, which consists of:
           - samples.tensors: batched images, of shape [batch_size x 3 x H x W]
           - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels, 
           - captions: 
           - encode_and_save:
           - memory_cache:
        
        Note: the behavior is not the same when in train/eval mode. In eval mode, language masking and vision masking is not applied

        It returns a dict with the following elements:
           - "pred_logits": the classification logits (including no-object) for all queries.
                            Shape= [batch_size x num_queries x (num_classes + 1)]
           - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                           (center_x, center_y, height, width). These values are normalized in [0, 1],
                           relative to the size of each individual image (disregarding possible padding).
                           See PostProcess for information on how to retrieve the unnormalized bounding box.
           - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                            dictionnaries containing the two above keys for emodel.transformer decoder layer.

        """
        # parsing samples into NestedTensor form if not already
        if not isinstance(samples, NestedTensor):
            samples = NestedTensor.from_tensor_list(samples)
        if one_pass:
            # this is only to create a more intuitive abstraction at inference time, where you only call the model one time 
            memory_cache = self(samples, captions, encode_and_save=True)
            outputs = self(samples, captions, encode_and_save=False, memory_cache=memory_cache)
            return outputs
        if encode_and_save:
            assert memory_cache is None
            assert not self.mvm_loss, "MVM is deprecated"
            assert not self.qa_dataset, "QA is deprecated"
            # img feats, img positional encodings
            # TODO: check if we need to deal with mask...
            # TODO: check if pooling breaks other staffs...
            raw_img_feats, img_pos = self.backbone(samples)
            src, mask = raw_img_feats[-1].decompose()
            masked_img_feats = src.clone()
            projected_img_feats = self.input_proj(masked_img_feats)
            # B, C, Hp, Wp
            ori_pro_img_feats = projected_img_feats.clone()
            batch_size = ori_pro_img_feats.shape[0]

            # constructing queries, decoder is all you need XD
            # first go by Batch, Length, HiddenDim
            # transpose to Length, BatchNum, HiddenDim at the end
            query_spans = []

            # the obj query embeddings
            this_obj_query = self.query_embed.weight.repeat(batch_size, 1, 1)
            query_embed = this_obj_query.clone()
            query_spans.append({"type": "obj_query", "span": this_obj_query.shape[1]})

            # pass the transformer cross-modal encoding part
            memory_cache = self.transformer(
                projected_img_feats,
                mask,
                query_embed,
                img_pos[-1],
                captions,
                encode_and_save=ModelStage.VL_Encode,
                text_memory=None,
                cross_memory=None,
                text_attention_mask=None,
                positive_map = positive_map
            )

            # NOTE: for the ACL version, language query is not appended at the end of object query 
            # if self.mlm_loss:
            #     transpoed_lang_query = memory_cache['text_memory_resized'].transpose(0, 1)
            #     query_embed = torch.cat([query_embed, transpoed_lang_query], 1)
            #     query_spans.append({"type": "text_query", "span": transpoed_lang_query.shape[1]})
            #     transpoed_lang_query = memory_cache['text_memory_resized'].transpose(0, 1)

            # if self.qa_dataset is not None:
            #     # if do qa, add qa-specific embeddings
            #     this_qa_query = self.qa_embed.weight.repeat(batch_size, 1, 1)
            #     query_embed = torch.cat([query_embed, this_qa_query], 1)
            #     query_spans.append({"type": "qa_query", "span": this_qa_query.shape[1]})
            query_embed = rearrange(query_embed, 'b l c -> l b c')

            # if do contrastive learning, project to the same dimension
            memory_cache['query_spans'] = query_spans
            memory_cache['query_embed'] = query_embed
            # if self.mvm_loss:
            #     memory_cache["mvm_raw_projected_flattened"] = flatten_img_feats
            # NOTE: mlm loss memory is stored within the transformer
            return memory_cache

        else:
            # given the encoder result, run decoder part
            assert memory_cache is not None
            assert self.mvm_loss==False, "mvm loss is deprecated"
            # though the query is only used in this decoder part, it is contrusted in the prior part
            # query is in order: object, img, text, qa
            hs = self.transformer(
                mask=memory_cache["mask"],
                query_embed=memory_cache["query_embed"],
                pos_embed=memory_cache["pos_embed"],
                encode_and_save=ModelStage.Obj_Decode,
                text_memory=memory_cache["text_memory_resized"],
                cross_memory=memory_cache["cross_memory"],
                text_attention_mask=memory_cache["text_attention_mask"],
            )
            decoded_obj_feats = hs[0] # B, L, C (5, 100, 256), K,V for the language denoising transformer
            lang_query_feats = memory_cache['text_memory'].transpose(0,1) # B, L, C (5, 18+, 256)
            lang_denoised_feats = self.transformer(query_embed=lang_query_feats, encode_and_save=ModelStage.Lang_Decode, cross_memory=decoded_obj_feats) # B, L, C (5, 18+, 256)
            memory_cache['text_memory'] = lang_denoised_feats.transpose(0,1)
            # print("decoder output shape: ", hs.shape)
            # print(memory_cache['query_spans'])
            out = {}
            img_query_start, lang_query_start, lang_query_end = 0, 0, 0
            for this_q_info in memory_cache['query_spans']:
                if this_q_info['type'] == 'obj_query':
                    img_query_start += this_q_info['span']
                    lang_query_start += this_q_info['span']
                    lang_query_end += this_q_info['span']
                if this_q_info['type'] == 'img_query':
                    lang_query_start += this_q_info['span']
                    lang_query_end += this_q_info['span']
                if this_q_info['type'] == 'text_query':
                    lang_query_end += this_q_info['span']
            if self.mlm_loss:
                out["mlm_logits"] = self.mlm_head(lang_denoised_feats)
            if not self.training:
                out["lang_embeddings"] = lang_denoised_feats.detach()
            # if self.mvm_loss:
            #     assert False, "mvm loss should not be used"
            #     # actually not useful
            #     out["mvm_recovered_img"] = self.mvm_head(hs[0, :, img_query_start:lang_query_start])
            # if self.qa_dataset is not None:
            #     if self.split_qa_heads:
            #         if self.qa_dataset == "gqa":
            #             answer_embeds = hs[0, :, -6:]
            #             hs = hs[:, :, :-6]
            #             out["pred_answer_type"] = self.answer_type_head(answer_embeds[:, 0])
            #             out["pred_answer_obj"] = self.answer_obj_head(answer_embeds[:, 1])
            #             out["pred_answer_rel"] = self.answer_rel_head(answer_embeds[:, 2])
            #             out["pred_answer_attr"] = self.answer_attr_head(answer_embeds[:, 3])
            #             out["pred_answer_cat"] = self.answer_cat_head(answer_embeds[:, 4])
            #             out["pred_answer_global"] = self.answer_global_head(answer_embeds[:, 5])
            #         elif self.qa_dataset == "clevr":
            #             answer_embeds = hs[0, :, -4:]
            #             hs = hs[:, :, :-4]
            #             out["pred_answer_type"] = self.answer_type_head(answer_embeds[:, 0])
            #             out["pred_answer_binary"] = self.answer_binary_head(answer_embeds[:, 1]).squeeze(-1)
            #             out["pred_answer_reg"] = self.answer_reg_head(answer_embeds[:, 2])
            #             out["pred_answer_attr"] = self.answer_attr_head(answer_embeds[:, 3])
            #         else:
            #             assert False, f"Invalid qa dataset {self.qa_dataset}"

            #     else:
            #         answer_embeds = hs[0, :, -1]
            #         hs = hs[:, :, :-1]
            #         out["pred_answer"] = self.answer_head(answer_embeds)

            hs = hs[:, :, :img_query_start]
            outputs_class = self.class_embed(hs)
            outputs_coord = self.bbox_embed(hs).sigmoid()
            out.update(
                {
                    "pred_logits": outputs_class[-1],
                    "pred_boxes": outputs_coord[-1],
                }
            )
            outputs_isfinal = None
            if self.isfinal_embed is not None:
                outputs_isfinal = self.isfinal_embed(hs)
                out["pred_isfinal"] = outputs_isfinal[-1]
            proj_queries, proj_tokens = None, None
            if self.contrastive_align_loss:
                proj_queries = F.normalize(self.contrastive_align_projection_image(hs), p=2, dim=-1)
                proj_tokens = F.normalize(
                    self.contrastive_align_projection_text(memory_cache["text_memory"]).transpose(0, 1), p=2, dim=-1
                )
                out.update(
                    {
                        "proj_queries": proj_queries[-1],
                        "proj_tokens": proj_tokens,
                        "tokenized": memory_cache["tokenized"],
                    }
                )
            if self.aux_loss:
                if self.contrastive_align_loss:
                    assert proj_tokens is not None and proj_queries is not None
                    out["aux_outputs"] = [
                        {
                            "pred_logits": a,
                            "pred_boxes": b,
                            "proj_queries": c,
                            "proj_tokens": proj_tokens,
                            "tokenized": memory_cache["tokenized"],
                        }
                        for a, b, c in zip(outputs_class[:-1], outputs_coord[:-1], proj_queries[:-1])
                    ]
                else:
                    # not used
                    out["aux_outputs"] = [
                        {
                            "pred_logits": a,
                            "pred_boxes": b,
                        }
                        for a, b in zip(outputs_class[:-1], outputs_coord[:-1])
                    ]
                if outputs_isfinal is not None:
                    assert len(outputs_isfinal[:-1]) == len(out["aux_outputs"])
                    for i in range(len(outputs_isfinal[:-1])):
                        out["aux_outputs"][i]["pred_isfinal"] = outputs_isfinal[i]
            return out



class QACriterionGQA(nn.Module):
    def __init__(self, split_qa_heads):
        super().__init__()
        self.split_qa_heads = split_qa_heads

    def forward(self, output, answers):
        loss = {}
        if not self.split_qa_heads:
            loss["loss_answer_total"] = F.cross_entropy(output["pred_answer"], answers["answer"], reduction="mean")
            attr_total = (output["pred_answer"].argmax(-1)) == answers["answer"]
            loss["accuracy_answer_total"] = attr_total.float().mean()
            return loss

        device = output["pred_answer_type"].device
        loss["loss_answer_type"] = F.cross_entropy(output["pred_answer_type"], answers["answer_type"])

        type_acc = output["pred_answer_type"].argmax(-1) == answers["answer_type"]
        loss["accuracy_answer_type"] = type_acc.sum() / answers["answer_type"].numel()

        is_obj = answers["answer_type"] == 0
        is_attr = answers["answer_type"] == 1
        is_rel = answers["answer_type"] == 2
        is_global = answers["answer_type"] == 3
        is_cat = answers["answer_type"] == 4

        ## OBJ type
        obj_norm = is_obj.sum() if is_obj.any() else 1.0
        loss["loss_answer_obj"] = (
            F.cross_entropy(output["pred_answer_obj"], answers["answer_obj"], reduction="none")
            .masked_fill(~is_obj, 0)
            .sum()
            / obj_norm
        )
        obj_acc = (output["pred_answer_obj"].argmax(-1)) == answers["answer_obj"]
        loss["accuracy_answer_obj"] = (
            obj_acc[is_obj].sum() / is_obj.sum() if is_obj.any() else torch.as_tensor(1.0, device=device)
        )

        ## ATTR type
        attr_norm = is_attr.sum() if is_attr.any() else 1.0
        loss["loss_answer_attr"] = (
            F.cross_entropy(output["pred_answer_attr"], answers["answer_attr"], reduction="none")
            .masked_fill(~is_attr, 0)
            .sum()
            / attr_norm
        )
        attr_acc = (output["pred_answer_attr"].argmax(-1)) == answers["answer_attr"]
        loss["accuracy_answer_attr"] = (
            attr_acc[is_attr].sum() / is_attr.sum() if is_attr.any() else torch.as_tensor(1.0, device=device)
        )

        ## REL type
        rel_norm = is_rel.sum() if is_rel.any() else 1.0
        loss["loss_answer_rel"] = (
            F.cross_entropy(output["pred_answer_rel"], answers["answer_rel"], reduction="none")
            .masked_fill(~is_rel, 0)
            .sum()
            / rel_norm
        )
        rel_acc = (output["pred_answer_rel"].argmax(-1)) == answers["answer_rel"]
        loss["accuracy_answer_rel"] = (
            rel_acc[is_rel].sum() / is_rel.sum() if is_rel.any() else torch.as_tensor(1.0, device=device)
        )

        ## GLOBAL type
        global_norm = is_global.sum() if is_global.any() else 1.0
        loss["loss_answer_global"] = (
            F.cross_entropy(output["pred_answer_global"], answers["answer_global"], reduction="none")
            .masked_fill(~is_global, 0)
            .sum()
            / global_norm
        )
        global_acc = (output["pred_answer_global"].argmax(-1)) == answers["answer_global"]
        loss["accuracy_answer_global"] = (
            global_acc[is_global].sum() / is_global.sum() if is_global.any() else torch.as_tensor(1.0, device=device)
        )

        ## CAT type
        cat_norm = is_cat.sum() if is_cat.any() else 1.0
        loss["loss_answer_cat"] = (
            F.cross_entropy(output["pred_answer_cat"], answers["answer_cat"], reduction="none")
            .masked_fill(~is_cat, 0)
            .sum()
            / cat_norm
        )
        cat_acc = (output["pred_answer_cat"].argmax(-1)) == answers["answer_cat"]
        loss["accuracy_answer_cat"] = (
            cat_acc[is_cat].sum() / is_cat.sum() if is_cat.any() else torch.as_tensor(1.0, device=device)
        )

        loss["accuracy_answer_total"] = (
            type_acc
            * (is_obj * obj_acc + is_rel * rel_acc + is_attr * attr_acc + is_global * global_acc + is_cat * cat_acc)
        ).sum() / type_acc.numel()

        return loss


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build(args):
    num_classes = 255
    device = torch.device(args.device)

    assert not args.masks or args.mask_model != "none"

    qa_dataset = None
    assert not args.do_qa, "Question answering is deprecated in this version" 
    # if args.do_qa:
    #     assert not (
    #         ("clevr" in args.combine_datasets or "clevr_question" in args.combine_datasets)
    #         and "gqa" in args.combine_datasets
    #     ), "training GQA and CLEVR simultaneously is not supported"
    #     assert (
    #         "clevr_question" in args.combine_datasets
    #         or "clevr" in args.combine_datasets
    #         or "gqa" in args.combine_datasets
    #     ), "Question answering require either gqa or clevr dataset"
    #     qa_dataset = "gqa" if "gqa" in args.combine_datasets else "clevr"

    backbone = build_backbone(args)

    transformer = build_transformer(args)

    model = MDETR(
        backbone,
        transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        aux_loss=args.aux_loss,
        contrastive_hdim=args.contrastive_loss_hdim,
        contrastive_align_loss=args.contrastive_align_loss,
        qa_dataset=qa_dataset,
        split_qa_heads=args.split_qa_heads,
        predict_final=args.predict_final,
        mlm_loss = args.mlm_loss,
        mvm_loss = args.mvm_loss,
        mvm_prob = args.mvm_prob,
        freeze_vision_encoder=args.freeze_vision_encoder,
    )
    if args.mask_model != "none":
        model = DETRsegm(
            model,
            mask_head=args.mask_model,
            freeze_detr=(args.frozen_weights is not None),
        )
    matcher = build_matcher(args)
    weight_dict = {"loss_ce": args.ce_loss_coef, "loss_bbox": args.bbox_loss_coef}
    if args.contrastive_align_loss:
        weight_dict["loss_contrastive_align"] = args.contrastive_align_loss_coef
    if args.predict_final:
        weight_dict["loss_isfinal"] = 1

    weight_dict["loss_giou"] = args.giou_loss_coef
    if args.masks:
        weight_dict["loss_mask"] = args.mask_loss_coef
        weight_dict["loss_dice"] = args.dice_loss_coef
    if args.mlm_loss:
        weight_dict["mlm_loss"] = args.mlm_loss_coef
    if args.mvm_loss:
        weight_dict["mvm_loss"] = args.mvm_loss_coef
    if args.do_qa:
        if args.split_qa_heads:
            weight_dict["loss_answer_type"] = 1 * args.qa_loss_coef
            if qa_dataset == "gqa":
                weight_dict["loss_answer_cat"] = 1 * args.qa_loss_coef
                weight_dict["loss_answer_attr"] = 1 * args.qa_loss_coef
                weight_dict["loss_answer_rel"] = 1 * args.qa_loss_coef
                weight_dict["loss_answer_obj"] = 1 * args.qa_loss_coef
                weight_dict["loss_answer_global"] = 1 * args.qa_loss_coef
            else:
                weight_dict["loss_answer_binary"] = 1
                weight_dict["loss_answer_attr"] = 1
                weight_dict["loss_answer_reg"] = 1

        else:
            weight_dict["loss_answer_total"] = 1 * args.qa_loss_coef

    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.obj_dec_layers - 1):
            aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ["labels", "boxes", "cardinality"]
    if args.masks:
        losses += ["masks"]
    if args.predict_final:
        losses += ["isfinal"]
    if args.contrastive_align_loss:
        losses += ["contrastive_align"]

    criterion = None
    if not args.no_detection:
        criterion = SetCriterion(
            num_classes,
            matcher=matcher,
            eos_coef=args.eos_coef,
            losses=losses,
            temperature=args.temperature_NCE,
        )
        criterion.to(device)

    contrastive_criterion = None

    if args.do_qa:
        if qa_dataset == "gqa":
            qa_criterion = QACriterionGQA(split_qa_heads=args.split_qa_heads)
        elif qa_dataset == "clevr":
            qa_criterion = QACriterionClevr()
        else:
            assert False, f"Invalid qa dataset {qa_dataset}"
        qa_criterion.to(device)
    else:
        qa_criterion = None

    if args.mvm_loss:
        # mvm_criterion = MaskVisionContrastiveCriterion(args.mvm_temp)
        mvm_criterion = MaskVisionContrastiveCriterion(0.07)
        mvm_criterion.to(device)
    else:
        mvm_criterion = None

    if args.mlm_loss:
        mlm_criterion = MaskLanguageCriterion()
        mlm_criterion.to(device)
    else:
        mlm_criterion = None
    return model, criterion, contrastive_criterion, qa_criterion, mvm_criterion, mlm_criterion, weight_dict
