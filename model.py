import torch
from torch import nn
import resnet


def generate_model(args):

    
    print('Without Pre-trained model')
    assert args.model_depth in [18, 50, 101]
    if args.model_depth == 18:

        model = resnet.resnet18(
            output_dim=args.feature_dim,
            sample_size=args.sample_size,
            sample_duration=args.sample_duration,
            num_classes=args.num_classes,
            shortcut_type=args.shortcut_type,
            tracking=args.tracking
        )

    elif args.model_depth == 50:
        model = resnet.resnet50(
            output_dim=args.feature_dim,
            sample_size=args.sample_size,
            sample_duration=args.sample_duration,
            num_classes=args.num_classes,
            shortcut_type=args.shortcut_type,
            tracking=args.tracking
        )

    elif args.model_depth == 101:
        model = resnet.resnet101(
            output_dim=args.feature_dim,
            sample_size=args.sample_size,
            sample_duration=args.sample_duration,
            num_classes=args.num_classes,
            shortcut_type=args.shortcut_type,
            tracking=args.tracking
        )

 #   model = nn.DataParallel(model, device_ids=None)
	

    return model

