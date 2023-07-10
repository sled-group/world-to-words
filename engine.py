# Copyright (c) Aishwarya Kamath & Nicolas Carion. Licensed under the Apache License 2.0. All Rights Reserved
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""
import math
import sys
from typing import Dict, Iterable, Optional

import torch
import torch.nn
import torch.optim

import utils.dist as dist
from datasets.clevrref import ClevrRefEvaluator
from datasets.coco_eval import CocoEvaluator
from datasets.flickr_eval import FlickrEvaluator
from datasets.phrasecut_eval import PhrasecutEvaluator
from datasets.refexp import RefExpEvaluator
from utils.metrics import MetricLogger, SmoothedValue
from utils.misc import targets_to
from utils.optim import adjust_learning_rate, update_ema
import wandb
from copy import deepcopy
from math import log10
from torch.cuda.amp import autocast

from pathlib import Path
from tqdm import tqdm

def train_one_epoch(
	model: torch.nn.Module,
	# note: contrastive align if part of the criterion
	criterion: Optional[torch.nn.Module],
	contrastive_criterion: Optional[torch.nn.Module],
	mvm_criterion: Optional[torch.nn.Module],
	mlm_criterion: Optional[torch.nn.Module],
	qa_criterion: Optional[torch.nn.Module],
	weight_dict: Dict[str, float],
	data_loader: Iterable,
	optimizer: torch.optim.Optimizer,
	device: torch.device,
	epoch: int,
	args,
	max_norm: float = 0,
	model_ema: Optional[torch.nn.Module] = None,
	grad_scalar = None,
	start_iter: int = 0,
):
	model.train()
	if criterion is not None:
		criterion.train()
	if contrastive_criterion is not None:
		contrastive_criterion.train()
	if qa_criterion is not None:
		qa_criterion.train()
	num_training_steps = int(len(data_loader) * args.epochs)
	for i, batch_dict in tqdm(enumerate(data_loader)):
		optimizer.zero_grad(set_to_none=True)
		with autocast():
			curr_step = epoch * len(data_loader) + i
			samples = batch_dict["samples"].to(device)
			positive_map = batch_dict["positive_map"].to(device) if "positive_map" in batch_dict else None
			targets = batch_dict["targets"]
			answers = {k: v.to(device) for k, v in batch_dict["answers"].items()} if "answers" in batch_dict else None
			captions = [t["caption"] for t in targets]

			targets = targets_to(targets, device)

			memory_cache = None
			if args.masks:
				# not used in MaskDETR
				outputs = model(samples, captions)
			else:
				memory_cache = model(samples, captions, encode_and_save=True, positive_map= [t['positive_map'] for t in targets])
				# TODO: Why cannot pass into it??
				temp_tokenized = memory_cache["tokenized"]
				outputs = model(samples, captions, encode_and_save=False, memory_cache=memory_cache)
				outputs['tokenized'] = temp_tokenized
				if outputs['aux_outputs']:
					for t in outputs['aux_outputs']:
						t['tokenized'] = temp_tokenized
			

			loss_dict = {}
			if criterion is not None:
				loss_dict.update(criterion(outputs, targets, positive_map))

			if contrastive_criterion is not None:
				# this is not used (from github issue)
				assert memory_cache is not None
				contrastive_loss = contrastive_criterion(memory_cache["text_pooled_op"], memory_cache["cross_pooled_op"])
				loss_dict["contrastive_loss"] = contrastive_loss

			if qa_criterion is not None:
				answer_losses = qa_criterion(outputs, answers)
				loss_dict.update(answer_losses)
			if mvm_criterion is not None:
				assert False, "mvm Not implemented"
				mvm_loss = mvm_criterion(outputs, memory_cache)
				loss_dict['mvm_loss'] = mvm_loss
			if mlm_criterion is not None:
				mlm_loss = mlm_criterion(outputs, memory_cache)
				loss_dict['mlm_loss'] = mlm_loss

			losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

			# reduce losses over all GPUs for logging purposes
			loss_dict_reduced = dist.reduce_dict(loss_dict)
			loss_dict_reduced_unscaled = {f"{k}_unscaled": v for k, v in loss_dict_reduced.items()}
			loss_dict_reduced_scaled = {k: v * weight_dict[k] for k, v in loss_dict_reduced.items() if k in weight_dict}
			losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

			loss_value = losses_reduced_scaled.item()

			log_dict = {k: v.detach() for k,v in loss_dict_reduced.items()}
			log_dict['loss_all_scaled'] = losses.detach()
			if args.gpu == 0:
				wandb.log({"train/":log_dict})

			if not math.isfinite(loss_value):
				print("Loss is {}, stopping training".format(loss_value))
				print(loss_dict_reduced)
				sys.exit(1)

		if not torch.isnan(losses):
			# gradient backpropagation
			if grad_scalar is not None:
				grad_scalar.scale(losses).backward()
			else:
				losses.backward()

			# gradient clipping
			if max_norm > 0:
				torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

			# optimize
			if grad_scalar is not None:
				grad_scalar.step(optimizer)
				grad_scalar.update()
			else:
				optimizer.step()

		else:
			wandb.alert(
				title="Loss is NAN", 
				text=f"loss is not a number, be aware! at {curr_step}"
			)


		if not args.no_vl_mapping_stage2:
			adjust_learning_rate(
				optimizer,
				epoch,
				curr_step,
				num_training_steps=num_training_steps,
				args=args,
			)
		if model_ema is not None:
			update_ema(model, model_ema, args.ema_decay)

		# metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
		# metric_logger.update(lr=optimizer.param_groups[0]["lr"])
		# metric_logger.update(lr_backbone=optimizer.param_groups[1]["lr"])
		# metric_logger.update(lr_text_encoder=optimizer.param_groups[2]["lr"])
		if args.save_for_aoa:
			this_iter_num = curr_step
			do_save = False
			if this_iter_num < 20:
				if this_iter_num % 2 == 0:
					do_save = True
			if this_iter_num < 100:
				if this_iter_num % 5 == 0:
					do_save = True
			if this_iter_num < 500:
				if this_iter_num % 10 == 0:
					do_save = True
			elif this_iter_num < 1000:
				if this_iter_num % 50 == 0:
					do_save = True
			elif this_iter_num < 10000:
				if this_iter_num % 500 == 0:
					do_save = True
			else:
				if this_iter_num % 10000 == 0:
					do_save = True
			if do_save:
				assert args.output_dir is not None
				if args.output_dir:
					print("Saving model for AOA")
					output_dir = Path(args.output_dir)
					checkpoint_paths = [output_dir /  "checkpoint.pth"]
					checkpoint_paths.append(output_dir / f"checkpoint_step{curr_step}.pth")
					for checkpoint_path in checkpoint_paths:
						dist.save_on_master(
							{
								"model": model.module.state_dict(),
								"model_ema": model_ema.state_dict() if args.ema else None,
								"optimizer": optimizer.state_dict(),
								"epoch": epoch,
								"curr_step": curr_step,
								"args": args,
							},
							checkpoint_path,
						)
		if args.learn_word_till_converge:
			this_iter_num = curr_step
			if this_iter_num == 50:
				assert args.output_dir is not None
				if args.output_dir:
					output_dir = Path(args.output_dir)
					config_name = args.dataset_config.split("/")[-1].split(".")[0]
					checkpoint_paths = []
					if not args.is_gl:
						# checkpoint_paths = [output_dir / f"{config_name}.pth"]
						checkpoint_paths = [args.output_dir_raw +"/" + f"{config_name}.pth"]
					else:
						checkpoint_paths = [args.output_dir_raw + "/" + f"{config_name}_gl.pth"]
					print("Saving final model after continue learning")
					print(checkpoint_paths)
					# if not args.is_gl:
					# 	checkpoint_paths += [args.output_dir_raw / f"{config_name}.pth"]
					# else:
					# 	checkpoint_paths += [args.output_dir_raw / f"{config_name}_gl.pth"]
					# checkpoint_paths.append(output_dir / f"checkpoint_step{curr_step}.pth")
					for checkpoint_path in checkpoint_paths:
						dist.save_on_master(
							{
								"model": model.module.state_dict(),
								"model_ema": model_ema.state_dict() if args.ema else None,
								"optimizer": optimizer.state_dict(),
								"epoch": epoch,
								"curr_step": curr_step,
								"args": args,
							},
							checkpoint_path,
						)
				exit()

		if args.learn_all_words:
			this_iter_num = curr_step
			if this_iter_num == 50*31:
				assert args.output_dir is not None
				if args.output_dir:
					output_dir = Path(args.output_dir)
					config_name = args.dataset_config.split("/")[-1].split(".")[0]
					checkpoint_paths = []
					if not args.is_gl:
						# checkpoint_paths = [output_dir / f"{config_name}.pth"]
						checkpoint_paths = [args.output_dir_raw +"/" + f"{config_name}.pth"]
					else:
						checkpoint_paths = [args.output_dir_raw + "/" + f"{config_name}_gl.pth"]
					print("Saving final model after continue learning")
					print(checkpoint_paths)
					# if not args.is_gl:
					# 	checkpoint_paths += [args.output_dir_raw / f"{config_name}.pth"]
					# else:
					# 	checkpoint_paths += [args.output_dir_raw / f"{config_name}_gl.pth"]
					# checkpoint_paths.append(output_dir / f"checkpoint_step{curr_step}.pth")
					for checkpoint_path in checkpoint_paths:
						dist.save_on_master(
							{
								"model": model.module.state_dict(),
								"model_ema": model_ema.state_dict() if args.ema else None,
								"optimizer": optimizer.state_dict(),
								"epoch": epoch,
								"curr_step": curr_step,
								"args": args,
							},
							checkpoint_path,
						)
				exit()

@torch.no_grad()
def evaluate(
	model: torch.nn.Module,
	criterion: Optional[torch.nn.Module],
	contrastive_criterion: Optional[torch.nn.Module],
	qa_criterion: Optional[torch.nn.Module],
	postprocessors: Dict[str, torch.nn.Module],
	weight_dict: Dict[str, float],
	data_loader,
	evaluator_list,
	device: torch.device,
	args,
):
	model.eval()
	if criterion is not None:
		criterion.eval()
	if contrastive_criterion is not None:
		contrastive_criterion.eval()
	if qa_criterion is not None:
		qa_criterion.eval()

	metric_logger = MetricLogger(delimiter="  ")
	header = "Test:"

	for batch_dict in metric_logger.log_every(data_loader, 10, header):
		samples = batch_dict["samples"].to(device)
		positive_map = batch_dict["positive_map"].to(device) if "positive_map" in batch_dict else None
		targets = batch_dict["targets"]
		answers = {k: v.to(device) for k, v in batch_dict["answers"].items()} if "answers" in batch_dict else None
		captions = [t["caption"] for t in targets]

		targets = targets_to(targets, device)

		memory_cache = None
		if args.masks:
			outputs = model(samples, captions)
		else:
			memory_cache = model(samples, captions, encode_and_save=True)
			outputs = model(samples, captions, encode_and_save=False, memory_cache=memory_cache)

		loss_dict = {}
		if criterion is not None:
			loss_dict.update(criterion(outputs, targets, positive_map))

		if contrastive_criterion is not None:
			assert memory_cache is not None
			contrastive_loss = contrastive_criterion(memory_cache["text_pooled_op"], memory_cache["img_pooled_op"])
			loss_dict["contrastive_loss"] = contrastive_loss

		if qa_criterion is not None:
			answer_losses = qa_criterion(outputs, answers)
			loss_dict.update(answer_losses)

		# reduce losses over all GPUs for logging purposes
		loss_dict_reduced = dist.reduce_dict(loss_dict)
		loss_dict_reduced_scaled = {k: v * weight_dict[k] for k, v in loss_dict_reduced.items() if k in weight_dict}
		loss_dict_reduced_unscaled = {f"{k}_unscaled": v for k, v in loss_dict_reduced.items()}
		metric_logger.update(
			loss=sum(loss_dict_reduced_scaled.values()),
			**loss_dict_reduced_scaled,
			**loss_dict_reduced_unscaled,
		)

		if not args.no_detection:
			orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
			results = postprocessors["bbox"](outputs, orig_target_sizes)
			if "segm" in postprocessors.keys():
				target_sizes = torch.stack([t["size"] for t in targets], dim=0)
				results = postprocessors["segm"](results, outputs, orig_target_sizes, target_sizes)

			flickr_res = [] if "flickr_bbox" in postprocessors.keys() else None
			if "flickr_bbox" in postprocessors.keys():
				image_ids = [t["original_img_id"] for t in targets]
				sentence_ids = [t["sentence_id"] for t in targets]
				items_per_batch_element = [t["nb_eval"] for t in targets]
				positive_map_eval = batch_dict["positive_map_eval"].to(device)
				flickr_results = postprocessors["flickr_bbox"](
					outputs, orig_target_sizes, positive_map_eval, items_per_batch_element
				)
				assert len(flickr_results) == len(image_ids) == len(sentence_ids)
				for im_id, sent_id, output in zip(image_ids, sentence_ids, flickr_results):
					flickr_res.append({"image_id": im_id, "sentence_id": sent_id, "boxes": output})

			phrasecut_res = None
			if "phrasecut" in postprocessors.keys():
				phrasecut_res = postprocessors["phrasecut"](results)
				assert len(targets) == len(phrasecut_res)
				for i in range(len(targets)):
					phrasecut_res[i]["original_id"] = targets[i]["original_id"]
					phrasecut_res[i]["task_id"] = targets[i]["task_id"]

			res = {target["image_id"].item(): output for target, output in zip(targets, results)}

			for evaluator in evaluator_list:
				if isinstance(evaluator, FlickrEvaluator):
					evaluator.update(flickr_res)
				elif isinstance(evaluator, PhrasecutEvaluator):
					evaluator.update(phrasecut_res)
				else:
					evaluator.update(res)

	# gather the stats from all processes
	metric_logger.synchronize_between_processes()
	print("Averaged stats:", metric_logger)
	for evaluator in evaluator_list:
		evaluator.synchronize_between_processes()

	refexp_res = None
	flickr_res = None
	phrasecut_res = None
	for evaluator in evaluator_list:
		if isinstance(evaluator, CocoEvaluator):
			evaluator.accumulate()
			evaluator.summarize()

		elif isinstance(evaluator, (RefExpEvaluator, ClevrRefEvaluator)):
			refexp_res = evaluator.summarize()
		elif isinstance(evaluator, FlickrEvaluator):
			flickr_res = evaluator.summarize()
		elif isinstance(evaluator, PhrasecutEvaluator):
			phrasecut_res = evaluator.summarize()

	# accumulate predictions from all images

	stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
	for evaluator in evaluator_list:
		if isinstance(evaluator, CocoEvaluator):
			if "bbox" in postprocessors.keys():
				stats["coco_eval_bbox"] = evaluator.coco_eval["bbox"].stats.tolist()
			if "segm" in postprocessors.keys():
				stats["coco_eval_masks"] = evaluator.coco_eval["segm"].stats.tolist()

	if refexp_res is not None:
		stats.update(refexp_res)

	if flickr_res is not None:
		stats["flickr"] = flickr_res

	if phrasecut_res is not None:
		stats["phrasecut"] = phrasecut_res

	return stats
