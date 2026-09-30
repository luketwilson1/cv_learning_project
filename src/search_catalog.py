from pathlib import Path

from src.embeddings import CATALOG_ROOT, find_similar_products


def main() -> None:
    """Run a catalog-search sanity check using one catalog crop."""
    query_path = next(CATALOG_ROOT.glob("*/*.jpg"))

    print(f"Query: {query_path}")

    matches = find_similar_products(query_path=query_path, top_k=5)

    for rank, (class_name, similarity, image_path) in enumerate(matches, start=1):
        print(f"{rank}. {class_name}: "f"similarity={similarity:.4f}, "f"source={Path(image_path).name}")

if __name__ == "__main__":
    main()