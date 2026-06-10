import os
import pandas as pd
import numpy as np
from PIL import Image

from tqdm import tqdm
import torch
import clip


class FeatureExtractor:
    def __init__(self, model_name="ViT-B/32"):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f'The device being used: {self.device.upper()}')
        self.model, self.preprocess = clip.load(model_name, device=self.device)

    def extract_embedding(self, image_path):
        image = Image.open(image_path).convert('RGB')
        image = self.preprocess(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            embedding = self.model.encode_image(image)
            embedding /= embedding.norm(dim=-1, keepdim=True)

        return embedding.cpu().numpy().flatten().astype('float32')


def build_file_map(image_root):
    file_map = {}
    valid_ext = {'.jpg', '.jpeg', '.png'}

    print('Scan folders with images')
    for root, _, files in os.walk(image_root):
        for file_name in files:
            ext = os.path.splitext(file_name)[1].lower()
            if ext in valid_ext:
                full_path = os.path.join(root, file_name)
                rel_path = os.path.relpath(full_path, image_root)
                rel_path = rel_path.replace("\\", "/")
                file_map[rel_path] = full_path

    print(f'Image files found: {len(file_map)}')
    return file_map


def building_embeddings(
    metadata_csv,
    image_root,
    output_metadata_csv=r'D:\final_project\metadata_valid_balanced.csv',
    output_embeddings_npy=r'D:\final_project\embeddings_balanced.npy',
    limit=None
):
    print('Read metadata')
    df = pd.read_csv(metadata_csv)

    if 'filename' not in df.columns:
        raise ValueError('there is no "filename" column in metadata csv')

    if limit is not None:
        df = df.head(limit).copy()
        print(f'Limit is on: {limit} lines')

    file_map = build_file_map(image_root)
    extractor = FeatureExtractor()

    print("\nExamples from metadata:")
    print(df["filename"].head(10).tolist())

    print("\nExamples from file_map:")
    print(list(file_map.keys())[:10])

    valid_rows = []
    embeddings = []
    not_found_count = 0
    not_found_files = []
    error_count = 0
    found_count = 0

    print('\nStart to build embeddings\n')

    for _, row in tqdm(df.iterrows(), total=len(df)):
        filename = str(row['filename']).strip()
        image_path = file_map.get(filename)

        if image_path is None:
            not_found_count += 1
            not_found_files.append(filename)
            continue

        found_count += 1

        try:
            embedding = extractor.extract_embedding(image_path)

            row_dict = row.to_dict()
            row_dict["image_path"] = image_path

            valid_rows.append(row_dict)
            embeddings.append(embedding)

        except Exception as e:
            error_count += 1
            print(f"Error {filename}: {e}")

    print(f"\nMatched files: {found_count}")

    if len(valid_rows) == 0:
        print("Not a single image could be processed.")
        return
    
    if not_found_files:
         not_found_df = pd.DataFrame({"missing_filename": not_found_files})
         not_found_df.to_csv(r"D:\final_project\missing_files.csv", index=False)
         print("Missing files list saved: D:\\final_project\\missing_files.csv")

    valid_df = pd.DataFrame(valid_rows)
    embeddings_array = np.array(embeddings, dtype="float32")

    print("\nSaving the results")
    valid_df.to_csv(output_metadata_csv, index=False)
    np.save(output_embeddings_npy, embeddings_array)

    print("\nDone")
    print(f"CSV: {output_metadata_csv}")
    print(f"NPY: {output_embeddings_npy}")
    print(f"Success processed: {len(valid_df)}")
    print(f"Not found files: {not_found_count}")
    print(f"Processing errors: {error_count}")
    print(f"Embeddings shape: {embeddings_array.shape}")


if __name__ == "__main__":
    metadata_csv = r"D:\final_project\metadata_balanced.csv"
    image_root = r"D:\final_project\wikiart_balanced"

    building_embeddings(
        metadata_csv=metadata_csv,
        image_root=image_root,
        output_metadata_csv=r"D:\final_project\metadata_valid_balanced.csv",
        output_embeddings_npy=r"D:\final_project\embeddings_balanced.npy",
        limit=None
    )