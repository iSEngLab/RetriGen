import pandas as pd
import argparse
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


def handle_new_dataset(query_source: str,
                       searched_target: str):
    substring = '"<AssertPlaceHolder>"'
    assert_placeholder_idx = query_source.find(substring)
    if assert_placeholder_idx != -1:
        split_idx = assert_placeholder_idx + len(substring)
        new_source = query_source[:split_idx] + f" /* {searched_target} */" + query_source[split_idx:]
        return new_source
    else:
        logger.info("do not found <AssertPlaceHolder>")

    return None


def main(args):
    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    df = pd.read_csv(args.result_file)
    dataset = {
        "source": [],
        "target": []
    }

    for idx, data in tqdm(df.iterrows(), total=len(df)):
        query_source = data["query_source"]
        query_target = data["query_target"]
        searched_target = data["codebase_target"]

        new_source = handle_new_dataset(query_source, searched_target)
        if new_source is not None:
            dataset["source"].append(new_source)
            dataset["target"].append(query_target)

    pd.DataFrame(dataset).to_csv(args.output_file, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_file", type=str)
    parser.add_argument("--output_file", type=str)

    args = parser.parse_args()
    main(args)
