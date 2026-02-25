from data.centernet_dataset import CenterNetDataset
from visualize import visualize_sample


def main():
    print("Start loading the Dataset")
    dataset = CenterNetDataset("../prepared_dataset")
    print("Dataset loaded")

    visualize_sample(dataset, index=0)


if __name__ == "__main__":
    main()
