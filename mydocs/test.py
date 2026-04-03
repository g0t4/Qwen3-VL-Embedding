# %%
# Cell 1: Imports & paths
import sys
import os
import subprocess
import numpy as np
import torch
from pathlib import Path
from pdf2image import convert_from_path

cwd = Path.cwd()
REPO_ROOT = cwd.parent
sys.path.insert(0, str(REPO_ROOT))

PDFS_DIR = cwd / "docs"
IMAGES_DIR = cwd / "page_images"
CACHE_FILE = cwd / "embeddings_cache.npz"
IMAGES_DIR.mkdir(exist_ok=True)

print(f"PDFs dir: {PDFS_DIR}")
print(f"PDFs found: {[p.name for p in sorted(PDFS_DIR.glob('*.pdf'))]}")

from src.models.qwen3_vl_embedding import Qwen3VLEmbedder

embedder = Qwen3VLEmbedder("Qwen/Qwen3-VL-Embedding-2B")
print("Embedder ready")

# %%
# Cell 3: Convert PDFs to page images (skips pages already saved)
def pdf_to_page_images(pdf_path: Path, output_dir: Path) -> list:
    pdf_name = pdf_path.stem
    images = convert_from_path(str(pdf_path))
    pages = []
    for page_idx, img in enumerate(images):
        img_path = output_dir / f"{pdf_name}__page{page_idx + 1:03d}.png"
        if not img_path.exists():
            img.save(str(img_path))
        pages.append({
            "pdf": pdf_path,
            "page": page_idx + 1,
            "image_path": img_path,
        })
    return pages

all_pages = []
for pdf_path in sorted(PDFS_DIR.glob("*.pdf")):
    pages = pdf_to_page_images(pdf_path, IMAGES_DIR)
    all_pages.extend(pages)
    print(f"{pdf_path.name}: {len(pages)} pages")

print(f"\nTotal pages: {len(all_pages)}")

# %%
# Cell 4: Generate (or load cached) document embeddings
def build_embeddings(pages, batch_size=4):
    all_embeddings = []
    for i in range(0, len(pages), batch_size):
        batch = pages[i:i + batch_size]
        inputs = [{"image": str(p["image_path"])} for p in batch]
        embs = embedder.process(inputs)
        all_embeddings.append(embs.cpu().float())
        print(f"  Embedded {min(i + batch_size, len(pages))}/{len(pages)}")
    return torch.cat(all_embeddings, dim=0).numpy()

cache_paths = [str(p["image_path"]) for p in all_pages]

if CACHE_FILE.exists():
    cache = np.load(CACHE_FILE, allow_pickle=True)
    if list(cache["paths"]) == cache_paths:
        doc_embeddings = cache["embeddings"]
        print(f"Loaded cached embeddings: {doc_embeddings.shape}")
    else:
        print("Cache mismatch (PDFs changed?), rebuilding...")
        doc_embeddings = build_embeddings(all_pages)
        np.savez(CACHE_FILE, embeddings=doc_embeddings, paths=np.array(cache_paths))
        print(f"Built and cached embeddings: {doc_embeddings.shape}")
else:
    print("No cache found, building embeddings...")
    doc_embeddings = build_embeddings(all_pages)
    np.savez(CACHE_FILE, embeddings=doc_embeddings, paths=np.array(cache_paths))
    print(f"Built and cached embeddings: {doc_embeddings.shape}")

# %%
# Cell 5: Query helper — run this cell after editing the query below
def query(text: str, k: int = 5, open_top: bool = True):
    q_emb = embedder.process([{"text": text}]).cpu().float().numpy()
    scores = (q_emb @ doc_embeddings.T)[0]
    top_k = np.argsort(scores)[-k:][::-1]

    print(f"\nQuery: {text!r}")
    print("-" * 60)
    for rank, idx in enumerate(top_k, 1):
        p = all_pages[idx]
        print(f"  {rank}. [{scores[idx]:.4f}]  {p['pdf'].name}  —  page {p['page']}")
        print(f"         {p['image_path']}")

    if open_top:
        best = all_pages[top_k[0]]
        print(f"\nOpening top result: {best['image_path'].name}")
        print("  ", best['image_path'])
        # subprocess.run(["open", str(best['image_path'])])
        subprocess.run(["/home/wes/.iterm2/imgcat", str(best['image_path'])])

    return top_k

# %%
# Cell 6: Run queries here — tweak and re-send just this cell
# results = query("production reports")
user_question = "what is my billing address"
top_k = query(user_question)
best = all_pages[top_k[0]]

# %% 

from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
import torch

vl_model = "Qwen/Qwen3-VL-2B-Thinking"
model = Qwen3VLForConditionalGeneration.from_pretrained(
    vl_model,
    dtype=torch.bfloat16,
    # attn_implementation="flash_attention_2",
    device_map="auto",
)

processor = AutoProcessor.from_pretrained(vl_model)

# %% 
user_question = "what is my billing address"

# messages = [
#     # {
#     #     "role": "system",
#     #     "content": [
#     #         {
#     #             "type": "text", 
#     #             "text": "Thinking: You are a helpful assistant. Analyze the provided image (top embedding result) and answer the user's query based on the content of the image.",
#     #         },
#     #     ],
#     # },
#     {
#         "role": "user",
#         "content": [
#             {
#                 "type": "image",
#                 "image": str(best['image_path']),
#             },
#             {
#                 "type": "text", 
#                 "text": user_question,
#             },
#         ],
#     }
# ]
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
            },
            {"type": "text", "text": "Describe this image."},
        ],
    }
]

# Preparation for inference
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt"
)
inputs = inputs.to(model.device)

# # TEST if chat template used ... by decoding it
processor.batch_decode(
    inputs.input_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
)

# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)

