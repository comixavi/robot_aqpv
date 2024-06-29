import os
import pandas as pd


def merge_excel_files(directory, output_file):
    all_data_frames = []

    for file in os.listdir(directory):
        if file.endswith('.xlsx'):
            file_path = os.path.join(directory, file)
            method_name = file.replace('.xlsx', '')
            df = pd.read_excel(file_path)

            if 'Criteria' in df.columns and 'Values' in df.columns:
                df.set_index('Criteria', inplace=True)
                df.rename(columns={'Values': method_name}, inplace=True)
                df.reset_index(inplace=True)

                if df['Criteria'].duplicated().any():
                    print(f"Duplicate criteria found in {file}.")
                    df.drop_duplicates(subset='Criteria', inplace=True)

                all_data_frames.append(df)

    if all_data_frames:
        from functools import reduce
        result_df = reduce(lambda left, right: pd.merge(left, right, on='Criteria', how='outer'), all_data_frames)

        full_path = os.path.join(directory, output_file)
        result_df.to_excel(full_path, index=False, engine='openpyxl')
        print(f"Data successfully written to {full_path}.")
    else:
        print("No Excel files were processed.")


if __name__ == '__main__':
    dr = './results'
    out_file = 'results_merged.xlsx'
    merge_excel_files(dr, out_file)
