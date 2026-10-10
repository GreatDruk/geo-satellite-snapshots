"""CLI entry point; training and application are separate explicit commands."""
import argparse
from pathlib import Path

from .training import PROJECT_ROOT, train_models


def main():
    parser = argparse.ArgumentParser(description='Final AGMFA + ResNet50 pipeline')
    commands = parser.add_subparsers(dest='command', required=True)
    train = commands.add_parser('train', help='Fresh 25-epoch refit on train+val')
    train.add_argument('--raw-dir', type=Path, default=PROJECT_ROOT / 'data/raw')
    train.add_argument('--run-id', required=True)
    train.add_argument('--model', choices=['both', 'agmfa', 'resnet50'], default='both')
    train.add_argument('--batch-size', type=int, default=16)
    train.add_argument('--imagenet-weights', type=Path, help='Original torchvision ResNet50 ImageNet V2 .pth for offline initialization')
    train.add_argument('--device', default='auto')
    predict = commands.add_parser('predict', help='Geometry-4 probabilities and fixed ensemble')
    predict.add_argument('--agmfa-checkpoint', type=Path, required=True)
    predict.add_argument('--resnet50-checkpoint', type=Path, required=True)
    predict.add_argument('--sample-csv', type=Path, required=True)
    predict.add_argument('--image-dir', type=Path, required=True)
    predict.add_argument('--output-dir', type=Path, required=True)
    predict.add_argument('--batch-size', type=int, default=8)
    predict.add_argument('--device', default='auto')
    args = parser.parse_args()
    if args.command == 'train':
        names = ('agmfa', 'resnet50') if args.model == 'both' else (args.model,)
        result = train_models(args.raw_dir, args.run_id, names=names, batch_size=args.batch_size,
                              device=args.device, imagenet_weights=args.imagenet_weights)
    else:
        from .inference import export_predictions
        result = export_predictions({'agmfa': args.agmfa_checkpoint, 'resnet50': args.resnet50_checkpoint},
                                    args.sample_csv, args.image_dir, args.output_dir,
                                    batch_size=args.batch_size, device=args.device)
    for name, path in result.items():
        print(f'{name}: {path}')


if __name__ == '__main__':
    main()
