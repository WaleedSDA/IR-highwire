import sys
import numpy as np
import ir_datasets

def main():
    print("Loading highwire dataset...")
    try:
        ds = ir_datasets.load("highwire/trec-genomics-2006")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)

    print("Iterating over docs to collect lengths...")
    lengths_words = []
    lengths_chars = []
    
    # We will process in batches and print progress
    count = 0
    for doc in ds.docs_iter():
        text = doc.default_text()
        if text:
            # Word count estimate by whitespace splitting
            words = len(text.split())
            chars = len(text)
            lengths_words.append(words)
            lengths_chars.append(chars)
        else:
            lengths_words.append(0)
            lengths_chars.append(0)
            
        count += 1
        if count % 20000 == 0:
            print(f"Processed {count} documents...")
            
    print(f"Finished processing {count} documents.")
    
    lengths_words = np.array(lengths_words)
    lengths_chars = np.array(lengths_chars)
    
    min_w = np.min(lengths_words)
    max_w = np.max(lengths_words)
    mean_w = np.mean(lengths_words)
    std_w = np.std(lengths_words)
    median_w = np.median(lengths_words)
    
    min_c = np.min(lengths_chars)
    max_c = np.max(lengths_chars)
    mean_c = np.mean(lengths_chars)
    std_c = np.std(lengths_chars)
    median_c = np.median(lengths_chars)
    
    print("\n--- Document Length Statistics (Words) ---")
    print(f"Min: {min_w}")
    print(f"Max: {max_w}")
    print(f"Mean: {mean_w:.2f}")
    print(f"Median: {median_w:.2f}")
    print(f"Std Dev: {std_w:.2f}")
    print(f"Coefficient of Variation (Std Dev / Mean): {std_w/mean_w:.4f}")
    
    print("\n--- Document Length Statistics (Characters) ---")
    print(f"Min: {min_c}")
    print(f"Max: {max_c}")
    print(f"Mean: {mean_c:.2f}")
    print(f"Median: {median_c:.2f}")
    print(f"Std Dev: {std_c:.2f}")
    
if __name__ == "__main__":
    main()
