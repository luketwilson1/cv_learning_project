from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torchvision.models import ResNet50_Weights, resnet50

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CATALOG_ROOT = PROJECT_ROOT / "data" / "catalog_crops"
INDEX_PATH = PROJECT_ROOT / "data" / "catalog_index.pt"

class ProductEmbedder:
    """Create normalized visual embeddings with pretrained ResNet-50."""

    def __init__(self, device: str = "auto") -> None:
        """Initialize the pretrained feature extractor.

        Args:
            device: PyTorch device to use. Use `"auto"` to select CUDA when
                available and otherwise use the CPU.
        """
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device)

        weights = ResNet50_Weights.IMAGENET1K_V2
        self.transform = weights.transforms()

        self.model = resnet50(weights=weights)
        self.model.fc = nn.Identity()

        self.model.to(self.device)
        self.model.eval()

    def embed_image(self, image_path: Path) -> torch.Tensor:
        """Create a normalized embedding for one image.

        Args:
            image_path: Path to the cropped product image.

        Returns:
            A one-dimensional, L2-normalized ResNet-50 embedding.
        """
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image_tensor = self.transform(image).unsqueeze(0)

        image_tensor = image_tensor.to(self.device)

        with torch.inference_mode():
            embedding = self.model(image_tensor)

        embedding = embedding.squeeze(0)
        embedding = torch.nn.functional.normalize(embedding, dim=0)

        return embedding.cpu()


def build_catalog_index(catalog_root: Path = CATALOG_ROOT, output_path: Path = INDEX_PATH) -> None:
    """Build and save an embedding index from product catalog crops.

    Args:
        catalog_root: Directory containing one subdirectory per product class.
        output_path: Destination for the serialized catalog index.

    Raises:
        ValueError: If the catalog does not contain any JPEG images.
    """
    embedder = ProductEmbedder()

    embeddings = []
    class_names = []
    image_paths = []

    for class_directory in sorted(catalog_root.iterdir()):
        if not class_directory.is_dir():
            continue

        for image_path in sorted(class_directory.glob("*.jpg")):
            embedding = embedder.embed_image(image_path)

            embeddings.append(embedding)
            class_names.append(class_directory.name)
            image_paths.append(str(image_path))

            print(f"Embedded {image_path}")

    if not embeddings:
        raise ValueError(f"No catalog images found in {catalog_root}")

    embedding_matrix = torch.stack(embeddings)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save({"embeddings": embedding_matrix, "class_names": class_names, "image_paths": image_paths}, output_path)

    print(f"Saved {len(embeddings)} embeddings to {output_path}")
    print(f"Embedding matrix shape: {embedding_matrix.shape}")


def find_similar_products(query_path: Path, index_path: Path = INDEX_PATH, top_k: int = 5) -> list[tuple[str, float, str]]:
    """Find the catalog images most similar to a query image.

    Args:
        query_path: Path to the cropped query image.
        index_path: Path to the saved catalog embedding index.
        top_k: Maximum number of matches to return.

    Returns:
        Ranked `(class_name, similarity, image_path)` matches.
    """
    recognizer = CatalogRecognizer(index_path=index_path)

    return recognizer.search(query_path=query_path, top_k=top_k)


class CatalogRecognizer:
    """Identify product crops using a saved embedding catalog."""

    def __init__(self, index_path: Path = INDEX_PATH, device: str = "auto") -> None:
        """Load the embedding model and catalog index once.

        Args:
            index_path: Path to the saved catalog embedding index.
            device: PyTorch device used to create query embeddings.

        Raises:
            FileNotFoundError: If the catalog index does not exist.
        """
        if not index_path.exists():
            raise FileNotFoundError(f"Catalog index not found: {index_path}")

        catalog = torch.load(index_path, map_location="cpu", weights_only=True)

        self.catalog_embeddings = catalog["embeddings"]
        self.class_names = catalog["class_names"]
        self.image_paths = catalog["image_paths"]

        self.embedder = ProductEmbedder(device=device)

    def search(
        self,
        query_path: Path,
        top_k: int = 5,
    ) -> list[tuple[str, float, str]]:
        """Find the catalog references most similar to a query crop.

        Args:
            query_path: Path to the cropped query image.
            top_k: Maximum number of matches to return.

        Returns:
            A list of `(class_name, similarity, image_path)` tuples ordered
            from highest to lowest similarity.

        Raises:
            ValueError: If `top_k` is less than one.
            FileNotFoundError: If the query image does not exist.
        """
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        if not query_path.exists():
            raise FileNotFoundError(
                f"Query image not found: {query_path}"
            )

        query_embedding = self.embedder.embed_image(query_path)
        similarities = self.catalog_embeddings @ query_embedding

        number_of_matches = min(top_k, len(similarities))
        scores, indices = torch.topk(
            similarities,
            k=number_of_matches,
        )

        matches = []

        for score, index in zip(scores, indices):
            catalog_index = index.item()

            matches.append(
                (
                    self.class_names[catalog_index],
                    score.item(),
                    self.image_paths[catalog_index],
                )
            )

        return matches

if __name__ == "__main__":
    build_catalog_index()