import torch
import torch.distributed
import torch.nn.functional as F
from torch import nn

import utils.dist as dist
from utils import box_ops
from utils.metrics import accuracy
from utils.misc import NestedTensor, interpolate
from utils.infonce import InfoNCE
from .segmentation import DETRsegm, dice_loss, sigmoid_focal_loss
from torch import einsum
from einops import rearrange

class QACriterionClevr(nn.Module):
	def __init__(self):
		super().__init__()

	def forward(self, output, answers):
		loss = {}
		loss["loss_answer_type"] = F.cross_entropy(output["pred_answer_type"], answers["answer_type"])

		type_acc = output["pred_answer_type"].argmax(-1) == answers["answer_type"]
		loss["accuracy_answer_type"] = type_acc.sum() / answers["answer_type"].numel()

		is_binary = answers["answer_type"] == 0
		is_attr = answers["answer_type"] == 1
		is_reg = answers["answer_type"] == 2

		binary_norm = is_binary.sum() if is_binary.any() else 1.0
		loss["loss_answer_binary"] = (
			F.binary_cross_entropy_with_logits(output["pred_answer_binary"], answers["answer_binary"], reduction="none")
			.masked_fill(~is_binary, 0)
			.sum()
			/ binary_norm
		)
		bin_acc = (output["pred_answer_binary"].sigmoid() > 0.5) == answers["answer_binary"]
		loss["accuracy_answer_binary"] = (
			bin_acc[is_binary].sum() / is_binary.sum() if is_binary.any() else torch.as_tensor(1.0)
		)

		reg_norm = is_reg.sum() if is_reg.any() else 1.0
		loss["loss_answer_reg"] = (
			F.cross_entropy(output["pred_answer_reg"], answers["answer_reg"], reduction="none")
			.masked_fill(~is_reg, 0)
			.sum()
			/ reg_norm
		)
		reg_acc = (output["pred_answer_reg"].argmax(-1)) == answers["answer_reg"]
		loss["accuracy_answer_reg"] = reg_acc[is_reg].sum() / is_reg.sum() if is_reg.any() else torch.as_tensor(1.0)

		attr_norm = is_attr.sum() if is_attr.any() else 1.0
		loss["loss_answer_attr"] = (
			F.cross_entropy(output["pred_answer_attr"], answers["answer_attr"], reduction="none")
			.masked_fill(~is_attr, 0)
			.sum()
			/ attr_norm
		)
		attr_acc = (output["pred_answer_attr"].argmax(-1)) == answers["answer_attr"]
		loss["accuracy_answer_attr"] = (
			attr_acc[is_attr].sum() / is_attr.sum() if is_attr.any() else torch.as_tensor(1.0)
		)

		loss["accuracy_answer_total"] = (
			type_acc * (is_binary * bin_acc + is_reg * reg_acc + is_attr * attr_acc)
		).sum() / type_acc.numel()

		return loss


class SetCriterion(nn.Module):
	"""This class computes the loss for DETR.
	The process happens in two steps:
		1) we compute hungarian assignment between ground truth boxes and the outputs of the model
		2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
	"""

	def __init__(self, num_classes, matcher, eos_coef, losses, temperature):
		"""Create the criterion.
		Parameters:
			num_classes: number of object categories, omitting the special no-object category
			matcher: module able to compute a matching between targets and proposals
			eos_coef: relative classification weight applied to the no-object category
			losses: list of all the losses to be applied. See get_loss for list of available losses.
		"""
		super().__init__()
		self.num_classes = num_classes
		self.matcher = matcher
		self.eos_coef = eos_coef
		self.losses = losses
		self.temperature = temperature
		empty_weight = torch.ones(self.num_classes + 1)
		empty_weight[-1] = self.eos_coef
		self.register_buffer("empty_weight", empty_weight)

	def loss_isfinal(self, outputs, targets, positive_map, indices, num_boxes):
		"""This loss is used in some referring expression dataset (specifically Clevr-REF+)
		It trains the model to predict which boxes are being referred to (ie are "final")
		Eg if the caption is "the cube next to the cylinder", MDETR will detect both the cube and the cylinder.
		However, the cylinder is an intermediate reasoning step, only the cube is being referred here.
		"""
		idx = self._get_src_permutation_idx(indices)
		src_isfinal = outputs["pred_isfinal"][idx].squeeze(-1)
		target_isfinal = torch.cat([t["isfinal"][i] for t, (_, i) in zip(targets, indices)], dim=0)

		loss_isfinal = F.binary_cross_entropy_with_logits(src_isfinal, target_isfinal, reduction="none")

		losses = {}
		losses["loss_isfinal"] = loss_isfinal.sum() / num_boxes
		acc = (src_isfinal.sigmoid() > 0.5) == (target_isfinal > 0.5)
		if acc.numel() == 0:
			acc = acc.sum()
		else:
			acc = acc.float().mean()
		losses["accuracy_isfinal"] = acc

		return losses

	def loss_labels(self, outputs, targets, positive_map, indices, num_boxes):
		"""Classification loss (NLL)
		targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
		"""

		logits = outputs["pred_logits"].log_softmax(-1)  # BS x (num_queries) x (num_tokens)

		src_idx = self._get_src_permutation_idx(indices)
		tgt_idx = []
		offset = 0
		for i, (_, tgt) in enumerate(indices):
			tgt_idx.append(tgt + offset)
			offset += len(targets[i]["boxes"])
		tgt_idx = torch.cat(tgt_idx)

		tgt_pos = positive_map[tgt_idx]
		target_sim = torch.zeros_like(logits)
		target_sim[:, :, -1] = 1
		target_sim[src_idx] = tgt_pos

		loss_ce = -(logits * target_sim).sum(-1)

		eos_coef = torch.full(loss_ce.shape, self.eos_coef, device=target_sim.device)
		eos_coef[src_idx] = 1

		loss_ce = loss_ce * eos_coef
		loss_ce = loss_ce.sum() / num_boxes

		losses = {"loss_ce": loss_ce}

		return losses

	def loss_contrastive_align(self, outputs, targets, positive_map, indices, num_boxes):
		bs = outputs["proj_queries"].shape[0]
		tokenized = outputs["tokenized"]
		normalized_text_emb = outputs["proj_tokens"]  # BS x (num_tokens) x hdim
		normalized_img_emb = outputs["proj_queries"]  # BS x (num_queries) x hdim

		logits = (
			torch.matmul(normalized_img_emb, normalized_text_emb.transpose(-1, -2)) / self.temperature
		)  # BS x (num_queries) x (num_tokens)

		# construct a map such that positive_map[k, i,j] = True iff query i is associated to token j in batch item k
		# For efficiency, the construction happens on CPU, then the whole matrix is transferred to GPU in one go.
		positive_map = torch.zeros(logits.shape, dtype=torch.bool)
		for i, ((idx_src, idx_tgt), tgt) in enumerate(zip(indices, targets)):
			if "tokens_positive" in tgt:
				cur_tokens = [tgt["tokens_positive"][j] for j in idx_tgt]
			else:
				cur_tokens = [tgt["tokens"][j] for j in idx_tgt]

			for j, tok_list in enumerate(cur_tokens):
				for (beg, end) in tok_list:
					beg_pos = tokenized.char_to_token(i, beg)
					end_pos = tokenized.char_to_token(i, end - 1)
					if beg_pos is None:
						try:
							beg_pos = tokenized.char_to_token(beg + 1)
							if beg_pos is None:
								beg_pos = tokenized.char_to_token(beg + 2)
						except:
							beg_pos = None
					if end_pos is None:
						try:
							end_pos = tokenized.char_to_token(end - 2)
							if end_pos is None:
								end_pos = tokenized.char_to_token(end - 3)
						except:
							end_pos = None
					if beg_pos is None or end_pos is None:
						continue

					assert beg_pos is not None and end_pos is not None
					positive_map[i, idx_src[j], beg_pos : end_pos + 1].fill_(True)

		positive_map = positive_map.to(logits.device)
		positive_logits = -logits.masked_fill(~positive_map, 0)
		negative_logits = logits  # .masked_fill(positive_map, -1000000)

		boxes_with_pos = positive_map.any(2)
		pos_term = positive_logits.sum(2)
		neg_term = negative_logits.logsumexp(2)

		nb_pos = positive_map.sum(2) + 1e-6

		box_to_token_loss = ((pos_term / nb_pos + neg_term)).masked_fill(~boxes_with_pos, 0).sum()

		tokens_with_pos = positive_map.any(1)
		pos_term = positive_logits.sum(1)
		neg_term = negative_logits.logsumexp(1)

		nb_pos = positive_map.sum(1) + 1e-6

		tokens_to_boxes_loss = ((pos_term / nb_pos + neg_term)).masked_fill(~tokens_with_pos, 0).sum()
		tot_loss = (box_to_token_loss + tokens_to_boxes_loss) / 2

		return {"loss_contrastive_align": tot_loss / num_boxes}

	@torch.no_grad()
	def loss_cardinality(self, outputs, targets, positive_map, indices, num_boxes):
		"""Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
		This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
		"""
		pred_logits = outputs["pred_logits"]
		device = pred_logits.device
		tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
		## Count the number of predictions that are NOT "no-object" (which is the last class)
		# normalized_text_emb = outputs["proj_tokens"]  # BS x (num_tokens) x hdim
		# normalized_img_emb = outputs["proj_queries"]  # BS x (num_queries) x hdim

		# logits = torch.matmul(
		#    normalized_img_emb, normalized_text_emb.transpose(-1, -2)
		# )  # BS x (num_queries) x (num_tokens)
		# card_pred = (logits[:, :, 0] > 0.5).sum(1)
		card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
		card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
		losses = {"cardinality_error": card_err}
		return losses

	def loss_boxes(self, outputs, targets, positive_map, indices, num_boxes):
		"""Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
		targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
		The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
		"""
		assert "pred_boxes" in outputs
		idx = self._get_src_permutation_idx(indices)
		src_boxes = outputs["pred_boxes"][idx]
		target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)

		loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none")

		losses = {}
		losses["loss_bbox"] = loss_bbox.sum() / num_boxes

		loss_giou = 1 - torch.diag(
			box_ops.generalized_box_iou(box_ops.box_cxcywh_to_xyxy(src_boxes), box_ops.box_cxcywh_to_xyxy(target_boxes))
		)
		losses["loss_giou"] = loss_giou.sum() / num_boxes
		return losses

	def loss_masks(self, outputs, targets, positive_map, indices, num_boxes):
		"""Compute the losses related to the masks: the focal loss and the dice loss.
		targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
		"""
		assert "pred_masks" in outputs

		src_idx = self._get_src_permutation_idx(indices)
		tgt_idx = self._get_tgt_permutation_idx(indices)

		src_masks = outputs["pred_masks"]

		# TODO use valid to mask invalid areas due to padding in loss
		target_masks, valid = NestedTensor.from_tensor_list([t["masks"] for t in targets]).decompose()
		target_masks = target_masks.to(src_masks)

		src_masks = src_masks[src_idx]
		# upsample predictions to the target size
		src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:], mode="bilinear", align_corners=False)
		src_masks = src_masks[:, 0].flatten(1)

		target_masks = target_masks[tgt_idx].flatten(1)

		losses = {
			"loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
			"loss_dice": dice_loss(src_masks, target_masks, num_boxes),
		}
		return losses

	def _get_src_permutation_idx(self, indices):
		# permute predictions following indices
		batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
		src_idx = torch.cat([src for (src, _) in indices])
		return batch_idx, src_idx

	def _get_tgt_permutation_idx(self, indices):
		# permute targets following indices
		batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
		tgt_idx = torch.cat([tgt for (_, tgt) in indices])
		return batch_idx, tgt_idx

	def get_loss(self, loss, outputs, targets, positive_map, indices, num_boxes, **kwargs):
		loss_map = {
			"labels": self.loss_labels,
			"cardinality": self.loss_cardinality,
			"boxes": self.loss_boxes,
			"masks": self.loss_masks,
			"isfinal": self.loss_isfinal,
			"contrastive_align": self.loss_contrastive_align,
		}
		assert loss in loss_map, f"do you really want to compute {loss} loss?"
		return loss_map[loss](outputs, targets, positive_map, indices, num_boxes, **kwargs)

	def forward(self, outputs, targets, positive_map):
		"""This performs the loss computation.
		Parameters:
			 outputs: dict of tensors, see the output specification of the model for the format
			 targets: list of dicts, such that len(targets) == batch_size.
					  The expected keys in each dict depends on the losses applied, see each loss' doc
		"""
		outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}

		# Retrieve the matching between the outputs of the last layer and the targets
		indices = self.matcher(outputs_without_aux, targets, positive_map)

		# Compute the average number of target boxes accross all nodes, for normalization purposes
		num_boxes = sum(len(t["labels"]) for t in targets)
		num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
		if dist.is_dist_avail_and_initialized():
			torch.distributed.all_reduce(num_boxes)
		num_boxes = torch.clamp(num_boxes / dist.get_world_size(), min=1).item()

		# Compute all the requested losses
		losses = {}
		for loss in self.losses:
			losses.update(self.get_loss(loss, outputs, targets, positive_map, indices, num_boxes))

		# In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
		if "aux_outputs" in outputs:
			for i, aux_outputs in enumerate(outputs["aux_outputs"]):
				indices = self.matcher(aux_outputs, targets, positive_map)
				for loss in self.losses:
					if loss == "masks":
						# Intermediate masks losses are too costly to compute, we ignore them.
						continue
					kwargs = {}
					l_dict = self.get_loss(loss, aux_outputs, targets, positive_map, indices, num_boxes, **kwargs)
					l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
					losses.update(l_dict)

		return losses

class ContrastiveCriterion(nn.Module):
	def __init__(self, temperature=0.1):
		super().__init__()
		self.temperature = temperature

	def forward(self, pooled_text, pooled_image):

		normalized_text_emb = F.normalize(pooled_text, p=2, dim=1)
		normalized_img_emb = F.normalize(pooled_image, p=2, dim=1)

		logits = torch.mm(normalized_img_emb, normalized_text_emb.t()) / self.temperature
		labels = torch.arange(logits.size(0)).to(pooled_image.device)

		loss_i = F.cross_entropy(logits, labels)
		loss_t = F.cross_entropy(logits.t(), labels)
		loss = (loss_i + loss_t) / 2.0
		return loss


class MaskLanguageCriterion(nn.Module):
	def __init__(self):
		super().__init__()
		# NOTE: hard coded token num for roberta-base tokenizer
		self.tk_num = 50265
	
	def forward(self, outputs, memory_cache):
		logits = outputs["mlm_logits"]
		unmask_info = memory_cache["text_unmasking_info"].to(logits.device)
		# print(logits.shape)
		# print(unmask_info.shape)
		loss = F.cross_entropy(logits.view(-1, self.tk_num), unmask_info.view(-1))
		if torch.sum(unmask_info) == -100 * torch.numel(unmask_info):
			print("all masked, loss = 0")
			loss = torch.tensor(0.0, device=loss.device)
		return loss

# class MaskVisionCriterion(nn.Module):
# 	def __init__(self):
# 		super().__init__()
# 		self.criterion = nn.MSELoss()
	
# 	def forward(self, outputs, memory_cache):
# 		recovered_img_feats = outputs['mvm_recovered_img']
# 		ori_img_feats = memory_cache['mvm_raw_projected_flattened']
# 		# print(recovered_img_feats.shape)
# 		# print(ori_img_feats.shape)
# 		loss = self.criterion(recovered_img_feats, ori_img_feats)
# 		return loss

class MaskVisionContrastiveCriterion(nn.Module):
	def __init__(self, mvm_temp):
		super().__init__()
		self.temp = mvm_temp
	
	def forward(self, outputs, memory_cache):
		recovered_img_feats = outputs['mvm_recovered_img']
		ori_img_feats = memory_cache['mvm_raw_projected_flattened']
		flattened_ori = rearrange(ori_img_feats, 'b l c -> (b l) c')
		flattened_rev = rearrange(recovered_img_feats, 'b l c -> (b l) c')
		normed_ori = F.normalize(flattened_ori, p=2, dim=1)
		normed_rev = F.normalize(flattened_rev, p=2, dim=1)

		sim = einsum('i d, j d -> i j', normed_ori, normed_rev) * self.temp 
		labels = torch.arange(flattened_ori.shape[0], device = ori_img_feats.device) 
		loss_t = F.cross_entropy(sim, labels) 
		loss_i = F.cross_entropy(sim.T, labels) 
		return (loss_t + loss_i) / 2. 


