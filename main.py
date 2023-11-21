import torch

model = torch.hub.load("WongKinYiu/yolov7", "custom", 'models/yolov7.pt')


def fetch_people(img_path):
    imgs = [img_path]  # batch of images

    # Inference
    results = model(imgs)
    results.show()
    df = results.pandas().xyxy[0]
    df = df.loc[df['name'] == 'person']
    people = []
    for index, row in df.iterrows():
        x = int(row['xmin'] + (row['xmax'] - row['xmin']) // 2)
        y = int(row['ymin'] + (row['ymax'] - row['ymin']) // 2)
        center = (x, y)
        box_min = (int(row['xmin']), int(row['ymin']))
        box_max = (int(row['xmax']), int(row['ymax']))
        people.append((box_min, box_max, center))

    # print(people)

    return people


if __name__ == "__main__":
    # Sample Image URL
    image = 'people1.jpg'
    people_list = fetch_people(image)
    print('number of people: ', len(people_list))

