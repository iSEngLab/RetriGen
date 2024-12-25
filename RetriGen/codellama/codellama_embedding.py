import requests
import pandas as pd
import multiprocessing
import argparse

from tqdm import tqdm
from typing import List


def request_embedding_multi(session: requests.session, prompts: List[str], job_id: int):
    results = []
    for prompt in tqdm(prompts, desc="request codellama job {}".format(job_id), total=len(prompts)):
        result = request_embedding(session, prompt)
        results.append(result)

    return results


def request_embedding(session: requests.session, prompt: str):
    data = {
        "model": "codellama:7b-code-fp16",
        "prompt": prompt
    }
    headers = {
        "Content-Type": "application/json"
    }
    response = session.post("http://localhost:11434/api/embeddings", headers=headers, json=data)
    if response.status_code != 200:
        print("network error: " + str(response.status_code) + ", msg:" + response.text)

    return {
        "source": prompt,
        "embedding": response.json()["embedding"]
    }


def get_embedding_multi(res_col: pd.Series, session):
    # 指定任务数量
    job_n = 10

    # 获取当前的 job
    row_series = pd.Series(range(0, len(res_col)), index=res_col)
    job_n = min(job_n, len(row_series.index))
    row_cut = pd.qcut(row_series, job_n, labels=range(0, job_n))
    data_list = []
    for i in range(0, job_n):
        data_list.append(list(row_cut[row_cut == i].index))

    with multiprocessing.Pool(job_n) as pool:
        jobs = []
        for i in range(0, len(data_list)):
            jobs.append(
                pool.apply_async(
                    func=request_embedding_multi,
                    kwds={'session': session, 'prompts': data_list[i], "job_id": i}))

        # 把每个进程的结果拼接起来
        res_list = []
        for job in jobs:
            part_res = job.get()
            res_list.extend(part_res)

    return res_list


def main(args):
    source = pd.read_csv(args.source_path)
    result_dict = {
        "source": [],
        "target": [],
        "embedding": []
    }
    session = requests.session()
    results = get_embedding_multi(source["source"], session)
    for result in results:
        result_dict["source"].append(result["source"])
        result_dict["embedding"].append(result["embedding"])
    result_dict["target"].extend(source["target"])

    session.close()
    pd.DataFrame(result_dict).to_csv(args.target_path, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_path", type=str)
    parser.add_argument("--target_path", type=str)
    args = parser.parse_args()

    main(args)
