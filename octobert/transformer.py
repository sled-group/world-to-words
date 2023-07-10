# Copyright (c) Aishwarya Kamath & Nicolas Carion. Licensed under the Apache License 2.0. All Rights Reserved
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
from typing import List, Optional
import code

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers import RobertaModel, RobertaTokenizerFast
from utils.masker import Masker
from copy import deepcopy

from enum import Enum

class ModelStage(Enum):
    VL_Encode = 1
    Obj_Decode = 2
    Lang_Decode = 3

class Transformer(nn.Module):
    def __init__(
        self,
        d_model=512,
        nhead=4,
        num_vl_encoder_layers=4,
        num_obj_decoder_layers=4,
        num_lang_decoder_layers=4,
        dim_feedforward=1024,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
        return_intermediate_dec=False,
        pass_pos_and_query=True,
        text_encoder_type="roberta-base",
        freeze_text_encoder=False,
        do_mlm=False,
        mlm_conc_noun_prob=0.5,
        mlm_other_prob=0.15,
    ):
        super().__init__()

        self.pass_pos_and_query = pass_pos_and_query

        vl_encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation, normalize_before)
        # vl_encoder_layer = torch.nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, activation=activation, norm_first=normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.vl_encoder = TransformerEncoder(vl_encoder_layer, num_vl_encoder_layers, encoder_norm)

        # detection_decoder_layer = torch.nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation, normalize_before)

        detection_decoder_layer = TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, activation=activation, normalize_before=normalize_before)

        detection_decoder_norm = nn.LayerNorm(d_model)
        self.detection_decoder = TransformerDecoder(
            detection_decoder_layer, num_obj_decoder_layers, detection_decoder_norm, return_intermediate=return_intermediate_dec
        )

        lang_decoder_norm = nn.LayerNorm(d_model)
        lang_decoder_layer = torch.nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, activation=activation, norm_first=normalize_before, batch_first=True)
        self.lang_decoder = torch.nn.TransformerDecoder(
            lang_decoder_layer, num_lang_decoder_layers, lang_decoder_norm)
        self._reset_parameters()

        # --- text encoder ---
        # NOTE: the default bert config is just bert-base-uncased
        self.text_encoder = RobertaModel.from_pretrained(text_encoder_type)
        self.tokenizer = RobertaTokenizerFast.from_pretrained(text_encoder_type, return_special_tokens_mask=True)
        self.masker = Masker(self.tokenizer, concrete_noun_prob=mlm_conc_noun_prob, other_prob=mlm_other_prob)

        if freeze_text_encoder:
            for p in self.text_encoder.parameters():
                p.requires_grad_(False)

        # --- miscs ---
        self.expander_dropout = 0.1
        config = self.text_encoder.config
        self.resizer = FeatureResizer(
            input_feat_size=config.hidden_size,
            output_feat_size=d_model,
            dropout=self.expander_dropout,
        )

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        src=None,
        mask=None,
        query_embed=None,
        pos_embed=None,
        text=None,
        encode_and_save=ModelStage.VL_Encode,
        text_memory=None,
        cross_memory=None,
        text_attention_mask=None,
        positive_map = None,
    ):
        if encode_and_save == ModelStage.VL_Encode:
            # if encode and return the memory

            # src is the visual feature map
            # flatten NxCxHxW to HWxNxC
            bs, c, h, w = src.shape
            src = src.flatten(2).permute(2, 0, 1)
            device = src.device
            pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
            # query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
            mask = mask.flatten(1)

            if self.pass_pos_and_query:
                tgt = torch.zeros_like(query_embed)
            else:
                src, tgt, query_embed, pos_embed = src + 0.1 * pos_embed, query_embed, None, None

            device = src.device
            if isinstance(text[0], str):
                # Encode the text, mask language modeling
                tokenized = self.tokenizer.batch_encode_plus(text, padding="longest", return_tensors='pt' )
                if self.training:
                    if positive_map is not None:
                        # batch, token_len. True if part of concrete noun
                        reduced_map = [t.sum(dim=0) != 0 for t in positive_map]
                        positive_masks = torch.stack(reduced_map)
                        positive_masks = positive_masks[:, :tokenized['input_ids'].shape[1]].to(device)
                        _, unmasking_info = self.masker.torch_mask_tokens(tokenized['input_ids'], concrete_noun_mask=positive_masks)
                    else:
                        _, unmasking_info = self.masker.torch_mask_tokens(tokenized['input_ids'])
                else:
                    unmasking_info = torch.ones_like(tokenized['input_ids'])*-100
                tokenized = tokenized.to(device)
                
                # The collator will do masking inplace, so just input the original tokenized is fine
                encoded_text = self.text_encoder(**deepcopy(tokenized))
                # Transpose memory because pytorch's attention expects sequence first
                text_memory = encoded_text.last_hidden_state.transpose(0, 1)
                # Invert attention mask that we get from huggingface because its the opposite in pytorch transformer
                text_attention_mask = tokenized.attention_mask.ne(1).bool()

                # Resize the encoder hidden states to be of the same d_model as the decoder
                text_memory_resized = self.resizer(text_memory)
            else:
                # The text is already encoded, use as is.
                # text_attention_mask, text_memory_resized, tokenized = text
                tokenized = text.to(device)
                encoded_text = self.text_encoder(**deepcopy(tokenized))
                text_memory = encoded_text.last_hidden_state.transpose(0, 1)
                text_attention_mask = tokenized.attention_mask.ne(1).bool()
                text_memory_resized = self.resizer(text_memory)
                unmasking_info = torch.ones_like(tokenized['input_ids'])*-100
                

            # Concat on the sequence dimension
            src = torch.cat([src, text_memory_resized], dim=0)
            # For mask, sequence dimension is second
            mask = torch.cat([mask, text_attention_mask], dim=1)
            # Pad the pos_embed with 0 so that the addition will be a no-op for the text tokens
            pos_embed = torch.cat([pos_embed, torch.zeros_like(text_memory_resized)], dim=0)

            cross_memory = self.vl_encoder(src, src_key_padding_mask=mask, pos=pos_embed)

            text_memory = cross_memory[-len(text_memory_resized) :]


            assert cross_memory.shape[1] == text_memory.shape[1] == tgt.shape[0]
            memory_cache = {
                # Resize text encoder output states that has the same d_model as the decoder
                "text_memory_resized": text_memory_resized,
                # the output of textual part of VL encoder, (Length, Batch, Dim)
                "text_memory": text_memory,
                # cross_modal output state
                "cross_memory": cross_memory,
                # the CLS output of the text encoder, (Batch, Dim)
                "text_pooled_op": None,
                # the CLS output of the cross-modal encoder, (Batch, Dim)
                "cross_pooled_op": None,  
                # mask for segmenting the image
                "mask": mask,
                # the attention mask for text stream, True for mask
                "text_attention_mask": text_attention_mask,
                # text unmasking info
                "text_unmasking_info": unmasking_info,
                # the position embedding 
                "pos_embed": pos_embed,
                # the transformer decoder query embedding
                "query_embed": query_embed,
                # the masked, tokenized text
                "tokenized": tokenized,
            }
            return memory_cache
        elif encode_and_save==ModelStage.Obj_Decode:
            # if pass the output of encoder + query, return the decoder output
            if self.pass_pos_and_query:
                tgt = torch.zeros_like(query_embed)
            else:
                src, tgt, query_embed, pos_embed = src + 0.1 * pos_embed, query_embed, None, None

            assert cross_memory.shape[1] == text_memory.shape[1] == tgt.shape[1]

            hs = self.detection_decoder(
                tgt,
                cross_memory,
                memory_key_padding_mask=mask,
                pos=pos_embed,
                query_pos=query_embed,
            )
            return hs.transpose(1, 2)
        elif encode_and_save==ModelStage.Lang_Decode:
            out = self.lang_decoder(tgt=query_embed, memory=cross_memory)
            return out
        else:
            raise NotImplementedError


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(
        self,
        src,
        mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):

        output = src

        for layer in self.layers:
            output = layer(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        """_summary_

        Args:
            tgt: input query
            memory: memory from encoder
            tgt_mask: self-attention attention mask 
            memory_mask: cross-attention attention mask with memory
            tgt_key_padding_mask: self-attention key padding mask
            memory_key_padding_mask: cross-attention key padding mask with memory
            pos (Optional[Tensor], optional): _description_. Defaults to None.
            query_pos (Optional[Tensor], optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        output = tgt

        intermediate = []

        
        for layer in self.layers:
            output = layer(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                pos=pos,
                query_pos=query_pos,
            )
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        src,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(
        self,
        src,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(
        self,
        src,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.cross_attn_mem = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # self.cross_attn_text = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        # self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        # self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        q = k = self.with_pos_embed(tgt, query_pos)

        # Self attention
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)


        # Cross attention to cross_modal memory
        tgt2 = self.cross_attn_mem(
            query=self.with_pos_embed(tgt, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        # FFN
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm4(tgt)
        return tgt

    def forward(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        if self.normalize_before:
            assert False, "not implemented yet"
        return self.forward_post(
            tgt,
            memory,
            tgt_mask,
            memory_mask,
            tgt_key_padding_mask,
            memory_key_padding_mask,
            pos,
            query_pos,
        )


class FeatureResizer(nn.Module):
    """
    This class takes as input a set of embeddings of dimension C1 and outputs a set of
    embedding of dimension C2, after a linear transformation, dropout and normalization (LN).
    """

    def __init__(self, input_feat_size, output_feat_size, dropout, do_ln=True):
        super().__init__()
        self.do_ln = do_ln
        # Object feature encoding
        self.fc = nn.Linear(input_feat_size, output_feat_size, bias=True)
        self.layer_norm = nn.LayerNorm(output_feat_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(self, encoder_features):
        x = self.fc(encoder_features)
        if self.do_ln:
            x = self.layer_norm(x)
        output = self.dropout(x)
        return output


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_transformer(args):
    return Transformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_vl_encoder_layers=args.vl_enc_layers,
        num_obj_decoder_layers=args.obj_dec_layers,
        num_lang_decoder_layers=args.lang_dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
        pass_pos_and_query=args.pass_pos_and_query,
        text_encoder_type=args.text_encoder_type,
        freeze_text_encoder=args.freeze_text_encoder,
        do_mlm=args.mlm_loss,
        mlm_conc_noun_prob= args.mlm_conc_noun_prob,
        mlm_other_prob= args.mlm_other_prob,
    )

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")
