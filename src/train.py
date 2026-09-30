from src.model import train_model


def main():
    train_model(
    data_path="data/data.yaml",
    epochs=100,
    image_size=640,
)


if __name__ == "__main__":
    main()

