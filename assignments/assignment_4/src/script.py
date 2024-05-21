from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
import os
import matplotlib.pyplot as plt
from PIL import Image, Image
import pandas as pd


def face_detection(data_path):
    mtcnn = MTCNN(keep_all=True)
    resnet = InceptionResnetV1(pretrained='casia-webface').eval()
    img = Image.open(data_path)
    boxes, _ = mtcnn.detect(img)
    if boxes is not None:
        boxes_shape = len(boxes)
    else:
        boxes_shape = 0
    return boxes_shape



def process(data_path):
    results = []
    output_path = os.path.join("out")

    for directory in sorted(os.listdir(data_path)):
        subfolder = os.path.join(data_path, directory)
        image_files = sorted(os.listdir(subfolder))

        for index, image in enumerate(image_files):
            image_path = os.path.join(subfolder, image)
            try: 
                data = face_detection(image_path)
                results.append({
                    'Image': image, 
                    'Newspaper': directory, 
                    'Year': image.split("-")[1],
                    'Faces': data
                }) 
            except Exception as error:
                print(f"Error processing {image}: {error}")
            print(f"{index} of {len(image_files)}")
    
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_path, "faces.csv"), index=False)

    return df



def save_csv(df, output_path):
    output_path = os.path.join("out")
    df['Decade'] = df['Year'].astype(int) // 10 * 10
    df['Has_faces'] = df['Faces'].apply(lambda x: 1 if x else 0)

    grouped_df = df.groupby(['Newspaper', 'Decade']).agg(
        Pages_with_faces_sum=('Has_faces', 'sum'),
        Total_pages=('Faces', 'count') 
    ).reset_index()

    grouped_df['Percentage_pages_with_faces'] = round((grouped_df['Pages_with_faces_sum'] / grouped_df['Total_pages']) * 100, 2)
    grouped_df = grouped_df.sort_values(by=['Decade', 'Newspaper'])

    grouped_df.to_csv(os.path.join(output_path, "grouped_faces.csv"), index=False)
    return grouped_df



def plot_percentage(grouped_df):
    plt.figure(figsize=(10, 6))
    for newspaper, data in grouped_df.groupby('Newspaper'):
        plt.plot(data['Decade'], data['Percentage_pages_with_faces'], label=newspaper, marker='o')

    plt.title('Percentage of Pages with Faces per Decade Across Newspapers')
    plt.xlabel('Decade')
    plt.ylabel('Percentage of Pages with Faces')
    plt.legend()
    plt.grid(True)
    plt.xticks(grouped_df['Decade'].unique())
    plt.tight_layout()
    plt.savefig("out/percentage_faces.png")



def main():
    data_path = os.path.join("in","images")

    output_path = os.path.join("out")
        
    df = process(data_path)
    grouped_df = save_csv(df, output_path)
    plot_percentage(grouped_df)

if __name__ == "__main__":
    main()
