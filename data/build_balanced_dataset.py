import os
import shutil
import pandas as pd

def extract_style_from_filename(filename: str) -> str:
   return str(filename).split('/')[0]

def build_balanced_dataset(
      input_csv: str,
      source_image_root: str,
      output_csv: str,
      output_image_root: str,
      selected_styles: list,
      sample_per_style: int = 2000,
      random_seed: int = 42 
):
   print('Reading metadata')
   df = pd.read_csv(input_csv)

   if 'filename' not in df.columns:
      raise ValueError("Column 'filename' not found in csv")
   
   print('Extracting style from filename')
   df['style'] = df['filename'].apply(extract_style_from_filename)

   print('\nAvailable style in csv: ')
   print(sorted(df['style'].dropna().unique()))

   print('\nFiltering selected styles')
   filtered_df = df[df['style'].isin(selected_styles)].copy()

   print('Rows after style filtering: ', len(filtered_df))

   print('\nCounts by style before sampling: ')
   print(filtered_df['style'].value_counts())

   print('\nSampling up to', sample_per_style, 'images per style (seed=', random_seed, ')')
   sampled_df = (
      filtered_df.groupby('style', group_keys=False)
      .apply(lambda x: x.sample(n=min(sample_per_style, len(x)), random_state=random_seed))
      .reset_index(drop=True)
   )
   
   sampled_df['style'] = sampled_df['filename'].apply(extract_style_from_filename)

   print('Final sampled dataset size:', len(sampled_df))

   print('\nCounts by style after sampling:', sampled_df['style'].value_counts())

   os.makedirs(output_image_root, exist_ok=True)

   copied_count = 0
   missing_files = []

   print('\nCopying images into new dataset folder')

   for _, row in sampled_df.iterrows():
      relative_path = str(row['filename']).replace('\\', '/')
      source_path = os.path.join(source_image_root, *relative_path.split('/'))
      target_path = os.path.join(output_image_root, *relative_path.split('/'))

      target_dir = os.path.dirname(target_path)
      os.makedirs(target_dir, exist_ok=True)

      if os.path.exists(source_path):
         shutil.copy2(source_path, target_path)
         copied_count += 1
      else:
         missing_files.append(relative_path)

   print('Copied files: ', copied_count)
   print('Missing files during copy: ', len(missing_files))

   if missing_files:
         sampled_df = sampled_df[~sampled_df['filename'].isin(missing_files)].copy()
         print('Rows left after removing missing files: ', len(sampled_df))
      
   sampled_df.to_csv(output_csv, index = False)
   print('\nBalanced csv saved: ', output_csv)

   if missing_files:
         missing_csv = os.path.join(os.path.dirname(output_csv), 'missing_files_balanced.csv')
         pd.DataFrame({'missing_filename': missing_files}).to_csv(missing_csv, index=False)
         print('Missing files list saved: ', missing_csv)

   print('Done')

if __name__ == '__main__':
    selected_styles = [
    "Art_Nouveau_Modern",
    "Baroque",
    "Cubism",
    "Early_Renaissance",
    "Expressionism",
    "Fauvism",
    "High_Renaissance",
    "Mannerism_Late_Renaissance",
    "Pointillism",
    "Pop_Art",
    "Rococo",
    "Romanticism",
    "Realism",
    "Impressionism",
    "Post_Impressionism"
]

    build_balanced_dataset(
        input_csv=r'D:\final_project\wikiart.csv',
        source_image_root=r'D:\final_project\wikiart',
        output_csv=r'D:\final_project\metadata_balanced.csv',
        output_image_root=r'D:\final_project\wikiart_balanced',
        selected_styles=selected_styles,
        sample_per_style=2000,
        random_seed=42
    )

         