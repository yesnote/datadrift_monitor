"""Build PAL image embedding caches with a Hugging Face Google ViT model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from methods.pal.embeddings import write_embedding_cache
from methods.common.paths import is_relative_to
from methods.pal.vit_embeddings import (
    DEFAULT_GOOGLE_VIT_MODEL,
    EMBEDDING_OUTPUTS,
    extract_vit_embeddings,
    metadata_payload,
    missing_image_paths,
    read_coco_image_paths,
    write_metadata,
)


def _resolve_path(value: Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def _assert_not_code_refs(path: Path) -> None:
    resolved = path.resolve()
    code_refs = (ROOT / 'code_refs').resolve()
    if is_relative_to(resolved, code_refs):
        raise ValueError('Refusing to use code_refs as PAL embedding input/output: %s' % path)


def _resolve_inputs(values):
    paths = [_resolve_path(value) for value in values]
    for path in paths:
        _assert_not_code_refs(path)
    return paths


def _default_metadata_path(output_path: Path) -> Path:
    return output_path.with_name(output_path.name + '.meta.json')


def _validate_cache_output_path(output_path: Path) -> None:
    if output_path.suffix.lower() not in ('.npy', '.json'):
        raise ValueError('PAL embedding cache output must use .npy or .json: %s'
                         % output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Build a PAL RCSP image embedding cache from COCO-style image JSONs.')
    parser.add_argument(
        '--ann-file',
        type=Path,
        action='append',
        required=True,
        help='COCO-style annotation JSON. May be passed multiple times.')
    parser.add_argument(
        '--image-root',
        type=Path,
        default=Path('data/VOCdevkit'),
        help='Root prepended to COCO images[*].file_name values.')
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('work_dirs/pal_embeddings/voc_google_vit_embeddings.npy'),
        help='Output cache path. Use .npy or .json.')
    parser.add_argument(
        '--metadata-output',
        type=Path,
        default=None,
        help='Optional metadata JSON path. Defaults to <output>.meta.json.')
    parser.add_argument(
        '--model-name',
        default=DEFAULT_GOOGLE_VIT_MODEL,
        help='Hugging Face ViT model id or local model directory.')
    parser.add_argument(
        '--batch-size',
        type=int,
        default=16,
        help='Images per ViT forward pass.')
    parser.add_argument(
        '--device',
        default='auto',
        help='Torch device, for example auto, cuda, cuda:0, or cpu.')
    parser.add_argument(
        '--embedding-output',
        choices=EMBEDDING_OUTPUTS,
        default='pooler',
        help='Image-level ViT output to cache.')
    parser.add_argument(
        '--no-normalize',
        action='store_true',
        help='Store raw vectors instead of L2-normalized vectors.')
    parser.add_argument(
        '--skip-missing',
        action='store_true',
        help='Skip missing image files instead of failing.')
    parser.add_argument(
        '--max-images',
        type=int,
        default=None,
        help='Optional cap for quick local checks. Do not use for reproduction.')
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Disable per-batch progress prints.')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    annotation_paths = _resolve_inputs(args.ann_file)
    image_root = _resolve_path(args.image_root)
    _assert_not_code_refs(image_root)
    output_path = _resolve_path(args.output)
    _assert_not_code_refs(output_path)
    _validate_cache_output_path(output_path)
    metadata_path = (
        _resolve_path(args.metadata_output)
        if args.metadata_output is not None
        else _default_metadata_path(output_path)
    )
    _assert_not_code_refs(metadata_path)

    records = read_coco_image_paths(annotation_paths, image_root)
    missing = missing_image_paths(records)
    if missing and not args.skip_missing:
        first = '\n'.join(str(path) for path in missing[:10])
        raise SystemExit(
            'Missing %d image file(s). First paths:\n%s\n'
            'Pass --skip-missing only for debugging, not reproduction.'
            % (len(missing), first))
    if missing and args.skip_missing:
        missing_set = set(missing)
        records = [record for record in records if record.path not in missing_set]

    if args.max_images is not None:
        if args.max_images <= 0:
            raise SystemExit('--max-images must be positive')
        records = records[:args.max_images]

    if not records:
        raise SystemExit('No images available for embedding extraction')

    embeddings = extract_vit_embeddings(
        records,
        model_name=args.model_name,
        batch_size=args.batch_size,
        device=args.device,
        embedding_output=args.embedding_output,
        normalize=not args.no_normalize,
        progress=not args.quiet,
    )
    write_embedding_cache(embeddings, output_path)

    first_vector = next(iter(embeddings.values()))
    payload = metadata_payload(
        records=records,
        annotation_paths=annotation_paths,
        image_root=image_root,
        output_path=output_path,
        model_name=args.model_name,
        batch_size=args.batch_size,
        device=args.device,
        embedding_output=args.embedding_output,
        normalize=not args.no_normalize,
        embedding_dim=len(first_vector),
        skipped_missing=len(missing) if args.skip_missing else 0,
    )
    write_metadata(payload, metadata_path)

    print('Wrote PAL ViT embeddings: %s' % output_path)
    print('Images: %d' % len(embeddings))
    print('Embedding dim: %d' % len(first_vector))
    print('Metadata: %s' % metadata_path)


if __name__ == '__main__':
    main()
