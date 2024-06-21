import pandas as pd
import re
import os
from results.aggregate_results_script import get_short_name


def extract_subscripts(value):
    """Extract subscript unicode characters from a string."""
    subscripts = ''.join(re.findall(r'[\u2070-\u209Cᵃ-ᶻ]', value))
    transformed = None
    if subscripts:
        subscript_map = {
            'ᵃ': 'a', 'ᵇ': 'b', 'ᶜ': 'c', 'ᵈ': 'd', 'ᵉ': 'e',
            'ᶠ': 'f', 'ᵍ': 'g', 'ʰ': 'h', 'ᶦ': 'i', 'ʲ': 'j',
            'ᵏ': 'k', 'ˡ': 'l', 'ᵐ': 'm', 'ⁿ': 'n', 'ᵒ': 'o',
            'ᵖ': 'p', 'ᵠ': 'q', 'ʳ': 'r', 'ˢ': 's', 'ᵗ': 't',
            'ᵘ': 'u', 'ᵛ': 'v', 'ʷ': 'w', 'ˣ': 'x', 'ʸ': 'y',
            'ᶻ': 'z', '₀': '0', '₁': '1', '₂': '2', '₃': '3',
            '₄': '4', '₅': '5', '₆': '6', '₇': '7', '₈': '8', '₉': '9'
        }
        # Replace subscript characters with their corresponding normal characters
        transformed = ''.join(subscript_map.get(char, char) for char in subscripts)

    return transformed if transformed else None


def process_data(filename):
    """Process the data from a file to extract models and their subscripts."""
    # Initialize list to store processed data
    results = []

    # Open the file and read line by line
    with open(filename, 'r') as file:
        i = 0
        for line in file:
            if i < 2:
                i += 1
                continue

            parts = line.split()
            if len(parts) < 3:
                continue
            full_model_name = parts[1]
            model_name = get_short_name(full_model_name)
            mrr = parts[2]
            ndcg = parts[3]

            # Extract subscripts
            mrr_subscripts = extract_subscripts(mrr)
            ndcg_subscripts = extract_subscripts(ndcg)

            # Append the results
            results.append((model_name, mrr_subscripts, ndcg_subscripts))

    # Create DataFrame
    df = pd.DataFrame(results, columns=["Model", "RR@10", "nDCG@10"])
    return df


datasets_names = ['passage', 'nfcorpus', 'hotpotqa', 'fiqa', 'quora', 'dbpedia-entity', 'scifact']


def main():
    for dataset_name in datasets_names:
        path = os.path.abspath(os.getcwd()) + "/results/significance_reports/"
        # dataset_name = "fiqa"

        df = process_data(path + dataset_name + '.txt')
        df.to_csv(path + 'extracted_signifiance_relations/' + dataset_name + '.csv', index=False)


if __name__ == '__main__':
    main()
