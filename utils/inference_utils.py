import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
from torchvision.ops import nms

img_transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b


def non_max_suppression(processed_outputs):
    new_bboxes = []
    token_ids_to_bboxs = defaultdict(list)

    # we only do nms for bboxs within the exact same region
    for bbox in processed_outputs['bboxes']:
        token_ids_str = str(bbox['label']['token_idxs'])
        token_ids_to_bboxs[token_ids_str].append(bbox)

    for token_ids_str, this_bboxes in token_ids_to_bboxs.items():
        bboxes = [bbox['bbox'] for bbox in this_bboxes]
        scores = [bbox['conf'] for bbox in this_bboxes]
        keep = nms(torch.tensor(bboxes), torch.tensor(scores), 0.5)
        new_bboxes.extend([this_bboxes[i] for i in keep])
    processed_outputs['bboxes'] = new_bboxes
    return processed_outputs

def post_processor(outputs, img, tokenizer, confidence=0.7):
    ret = {}

    denoised_token_ids = outputs['mlm_logits'].argmax(-1)[0]
    # the first and last tokens are always <s> and </s>
    ret['cap'] = tokenizer.decode(denoised_token_ids[1:-1])

    probas = 1 - outputs['pred_logits'].float().softmax(-1)[0, :, -1].cpu()
    keep = (probas > confidence).cpu()

    bboxes = rescale_bboxes(outputs['pred_boxes'].float()[0, keep].cpu(), img.size)
    scores = probas[keep]

    positive_tokens = (outputs["pred_logits"].float()[0, keep].softmax(-1).cpu() > 0.1).nonzero().tolist()
    predicted_spans = defaultdict(lambda: {"text": "", "token_idxs": []})
    for tok in positive_tokens:
        item, pos = tok
        if pos >= 255 or pos in [0, len(denoised_token_ids) - 1]:
            # the first and last tokens are always <s> and </s>
            continue
        predicted_spans[item]['text'] += " " + tokenizer.decode([denoised_token_ids[pos]]).strip()
        predicted_spans[item]['token_idxs'].append(pos)
    labels = [predicted_spans[k] for k in sorted(list(predicted_spans.keys()))]

    ret['bboxes'] = []
    for bbox, score, label in zip(bboxes.tolist(), scores.tolist(), labels):
        ret['bboxes'].append({"bbox": bbox, "conf": score, "label": label})
    return non_max_suppression(ret)



def plot_results(img, processed_outputs, save_path=None):
    COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],[0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]
    plt.figure(figsize=(16,10))
    np_image = np.array(img)
    ax = plt.gca()
    for bbox in processed_outputs['bboxes']:
        (xmin, ymin, xmax, ymax) = bbox['bbox']
        l = bbox['label']['text']
        s = bbox['conf']
        c = COLORS[hash(l) % len(COLORS)]
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, color=c, linewidth=3))
        text = f'{l}: {s:0.2f}'
        ax.text(xmin, ymin, text, fontsize=15, bbox=dict(facecolor='white', alpha=0.8))

    plt.imshow(np_image)
    plt.tight_layout()
    plt.axis('off')
    if save_path is not None:
        if save_path == "numpy_array":
            import io
            io_buf = io.BytesIO()
            DPI = 100
            plt.savefig(io_buf, format='raw', dpi=DPI)
            io_buf.seek(0)
            img_arr = np.reshape(np.frombuffer(io_buf.getvalue(), dtype=np.uint8), newshape=(int(plt.gcf().bbox.bounds[3]), int(plt.gcf().bbox.bounds[2]), -1))
            io_buf.close()
            plt.close()
            return img_arr
        else:
            plt.savefig(save_path)
            plt.close()
        
    else:
        plt.show()
        plt.close()