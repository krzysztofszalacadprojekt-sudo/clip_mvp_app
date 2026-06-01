import time
from pathlib import Path
import sys

# Add the app directory to the Python path
sys.path.append(str(Path(__file__).parent))

from app.embedding_store import create_and_save_embeddings

def run_performance_test():
    """
    Measures the time it takes to create embeddings for 1000 images.
    """
    print("Starting performance test...")
    
    # Get all image paths and take the first 1000
    image_paths = list(Path("images").rglob("*.jpg"))
    image_paths.extend(list(Path("images").rglob("*.webp")))
    image_paths.extend(list(Path("images").rglob("*.png")))
    image_paths.extend(list(Path("images").rglob("*.jpeg")))


    if len(image_paths) < 1000:
        print(f"Warning: Found only {len(image_paths)} images, which is less than 1000.")
        test_image_paths = image_paths
    else:
        test_image_paths = image_paths[:1000]

    print(f"Testing with {len(test_image_paths)} images.")

    start_time = time.time()
    
    create_and_save_embeddings(image_paths=[str(p) for p in test_image_paths])
    
    end_time = time.time()
    
    elapsed_time = end_time - start_time
    
    print(f"Time taken to embed {len(test_image_paths)} images: {elapsed_time:.2f} seconds.")
    
    if len(test_image_paths) > 0:
        time_per_image = elapsed_time / len(test_image_paths)
        print(f"Average time per image: {time_per_image * 1000:.2f} ms.")
    
        estimated_time_50k = time_per_image * 50000
        print(f"Estimated time for 50,000 images: {estimated_time_50k / 60:.2f} minutes.")

if __name__ == "__main__":
    run_performance_test()
