"""A copy of the GGUF with only the first N blocks, so that llama-server's last hidden state is layer N's.

llama.cpp and vLLM return only the last layer's hidden state; a model cut after block N returns block
N's output after the final RMSNorm, which a linear head standardises anyway. Metadata is
copied as gguf's gguf_new_metadata copies it, with block_count set to N; the embeddings, the final norm
and the output matrix stay. The layer pattern of this model (full attention every fourth block) depends
only on the block index, so a cut keeps it.

    python scripts/truncate_gguf.py --layers 20
"""

import argparse
from pathlib import Path

import gguf

from guardian.env import setup

SOURCE = Path("D:/VScode_projects/assets/llm/qwen3.6-35b-a3b/Qwen3.6-35B-A3B-Q4_K_M.gguf")


def truncate(source: Path, layers: int, out: Path) -> None:
    reader = gguf.GGUFReader(source)
    arch = reader.fields[gguf.Keys.General.ARCHITECTURE].contents()
    count_key = f"{arch}.block_count"
    total = reader.fields[count_key].contents()
    if not 0 < layers < total:
        raise ValueError(f"layers must be between 1 and {total - 1}")
    writer = gguf.GGUFWriter(out, arch, endianess=reader.endianess)
    for field in reader.fields.values():
        if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF."):
            continue
        kind = field.types[0]
        sub = field.types[-1] if kind == gguf.GGUFValueType.ARRAY else None
        value = layers if field.name == count_key else field.contents()
        writer.add_key_value(field.name, value, kind, sub_type=sub)

    def kept(name: str) -> bool:
        return not name.startswith("blk.") or int(name.split(".")[1]) < layers

    tensors = [t for t in reader.tensors if kept(t.name)]
    for t in tensors:
        writer.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    for t in tensors:
        writer.write_tensor_data(t.data, tensor_endianess=reader.endianess)
    writer.close()


def main() -> None:
    setup()
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, default=SOURCE)
    ap.add_argument("--layers", type=int, required=True)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    out = args.out or args.source.parent / "layers" / f"{args.source.stem}-L{args.layers}.gguf"
    out.parent.mkdir(parents=True, exist_ok=True)
    truncate(args.source, args.layers, out)
    print(out, f"{out.stat().st_size / 2**30:.1f} GiB")


if __name__ == "__main__":
    main()
