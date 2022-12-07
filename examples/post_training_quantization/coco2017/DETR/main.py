import argparse
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import detr.util.misc as utils
import sys
sys.path.append("./detr")
from detr.datasets import get_coco_api_from_dataset
from val_transform_datasets import build_dataset
from model import build
import onnx
import onnx_graphsurgeon as gs

from sparsebit.quantization import QuantModel, parse_qconfig

from evaluation import evaluate

parser = argparse.ArgumentParser(description="PyTorch ImageNet Training")
parser.add_argument("qconfig", help="the path of quant config")
parser.add_argument(
    "-a",
    "--arch",
    metavar="ARCH",
    default="deit_tiny_patch16_224",
    help="ViT model architecture. (default: deit_tiny)",
)
parser.add_argument(
    "-j",
    "--num_workers",
    default=2,
    type=int,
    metavar="N",
    help="number of data loading workers (default: 4)",
)
parser.add_argument(
    "-b",
    "--batch-size",
    default=1,
    type=int,
    metavar="N",
    help="mini-batch size (default: 64), this is the total "
    "batch size of all GPUs on the current node when "
    "using Data Parallel or Distributed Data Parallel",
)
parser.add_argument(
    "-p",
    "--print-freq",
    default=10,
    type=int,
    metavar="N",
    help="print frequency (default: 10)",
)

# * Backbone
parser.add_argument('--backbone', default='resnet50', type=str,
                    help="Name of the convolutional backbone to use")
parser.add_argument('--dilation', action='store_true',
                    help="If true, we replace stride with dilation in the last convolutional block (DC5)")
parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                    help="Type of positional embedding to use on top of the image features")


# * Transformer
parser.add_argument('--enc_layers', default=6, type=int,
                    help="Number of encoding layers in the transformer")
parser.add_argument('--dec_layers', default=6, type=int,
                    help="Number of decoding layers in the transformer")
parser.add_argument('--dim_feedforward', default=2048, type=int,
                    help="Intermediate size of the feedforward layers in the transformer blocks")
parser.add_argument('--hidden_dim', default=256, type=int,
                    help="Size of the embeddings (dimension of the transformer)")
parser.add_argument('--dropout', default=0.1, type=float,
                    help="Dropout applied in the transformer")
parser.add_argument('--nheads', default=8, type=int,
                    help="Number of attention heads inside the transformer's attentions")
parser.add_argument('--num_queries', default=100, type=int,
                    help="Number of query slots")
parser.add_argument('--pre_norm', action='store_true')

# Loss
parser.add_argument('--aux_loss', dest='aux_loss', action='store_true',
                    help="Enables auxiliary decoding losses (loss at each layer)")
# * Matcher
parser.add_argument('--set_cost_class', default=1, type=float,
                    help="Class coefficient in the matching cost")
parser.add_argument('--set_cost_bbox', default=5, type=float,
                    help="L1 box coefficient in the matching cost")
parser.add_argument('--set_cost_giou', default=2, type=float,
                    help="giou box coefficient in the matching cost")
# * Loss coefficients
parser.add_argument('--mask_loss_coef', default=1, type=float)
parser.add_argument('--dice_loss_coef', default=1, type=float)
parser.add_argument('--bbox_loss_coef', default=5, type=float)
parser.add_argument('--giou_loss_coef', default=2, type=float)
parser.add_argument('--eos_coef', default=0.1, type=float,
                    help="Relative classification weight of the no-object class")

#configs for coco dataset
parser.add_argument('--dataset_file', default='coco')
parser.add_argument('--coco_path', type=str)
parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")
parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    
parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')

def main():
    args = parser.parse_args()
    device = args.device

    # get pretrained model from https://github.com/facebookresearch/detr
    model = torch.hub.load('facebookresearch/detr:main', 'detr_resnet50', pretrained=True)
    model, criterion, postprocessors = build(args, model)

    qconfig = parse_qconfig(args.qconfig)
    qmodel = QuantModel(model, config=qconfig).to(device)

    cudnn.benchmark = True

    dataset_val = build_dataset(image_set='val', args=args)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    data_loader_val = torch.utils.data.DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)
    base_ds = get_coco_api_from_dataset(dataset_val)

    dataset_calib = build_dataset(image_set='train', args=args)
    sampler_calib = torch.utils.data.RandomSampler(dataset_calib)
    data_loader_calib = torch.utils.data.DataLoader(dataset_calib, args.batch_size, sampler=sampler_calib,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)


    qmodel.eval()
    with torch.no_grad():
        qmodel.prepare_calibration()
        # forward calibration-set
        calibration_size = 16
        cur_size = 0
        for samples, _ in data_loader_calib:
            sample = samples.tensors.to(device)
            qmodel(sample)
            cur_size += args.batch_size
            if cur_size >= calibration_size:
                break
        qmodel.calc_qparams()
    qmodel.set_quant(w_quant=True, a_quant=True)

    test_stats, coco_evaluator = evaluate(qmodel, criterion, postprocessors,
                                              data_loader_val, base_ds, device, args.output_dir)

    qmodel.export_onnx(torch.randn(1, 3, 800, 1200), name="qDETR.onnx")

    # graph = gs.import_onnx(onnx.load("qDETR.onnx"))
    # Reshapes = [node for node in graph.nodes if node.op == "Reshape"]
    # for node in Reshapes:
    #     if isinstance(node.inputs[1], gs.Constant):
    #         if node.inputs[1].values[1]==7600:
    #             node.inputs[1].values[1] = 8
    #         elif node.inputs[1].values[1]==950:
    #             node.inputs[1].values[1] = 1
    #         elif node.inputs[1].values[1]==100:
    #             node.inputs[1].values[1] = 1
    #         elif node.inputs[1].values[1]==800:
    #             node.inputs[1].values[1] = 8

    # onnx.save(gs.export_onnx(graph), "qDETR.onnx")




if __name__ == "__main__":
    main()
