import os
from dotenv import load_dotenv

load_dotenv('.huggingface_env2')
print(os.environ['HF_HOME'])

from utils.common import ignore_warnings

import psutil
import time
import faiss
from faiss.contrib.datasets import DatasetSIFT1M
import numpy as np


def prepare_dataset():
    os.system("wget ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz && tar -xf sift.tar.gz")
    os.system(f"mkdir data/sift1M -p")
    os.system(f"mv sift/* data/sift1M")
    os.system("rm -rf sift/")
    os.system("rm sift.tar.gz")


def get_memory_usage_mb():
    process = psutil.Process()
    memory_info = process.memory_info()
    return memory_info.rss / (1024 * 1024)


def check_dataset_folder():
    folder_path = './data/sift1M'  # 확인하려는 폴더 경로
    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        if os.listdir(folder_path):  # 폴더 안에 파일이나 폴더가 있는지 확인
            print(f"'{folder_path}' 폴더가 존재하며, 파일이 있습니다.")
            return True
        else:
            print(f"'{folder_path}' 폴더가 존재하지만, 비어 있습니다.")
            return False
    else:
        print(f"'{folder_path}' 폴더가 존재하지 않습니다.")
        return False


if __name__ == '__main__':
    ignore_warnings()

    print("## 예제 12.2 실습 데이터 불러오기")
    if not check_dataset_folder():
        prepare_dataset()  # ./data/sift1M 에 저장됨

    ds = DatasetSIFT1M()  # ./data/sift1M 에서 auto로 데이터 load 함

    xq = ds.get_queries()  # 검색 할 데이터
    xb = ds.get_database()  # 저장된 벡터 데이터
    gt = ds.get_groundtruth()  # 실제 정답

    print("\n## 예제 12.3 데이터가 늘어날 때 색인/검색 시간, 메모리 사용량 변화")
    k = 1
    d = xq.shape[1]  # 데이터 임베딩 차원
    print("d: ", d)
    nq = 1000
    xq = xq[:nq]

    for i in range(1, 10, 2):
        start_memory = get_memory_usage_mb()
        start_indexing = time.time()
        index = faiss.IndexFlatL2(d)
        index.add(xb[:(i + 1) * 100000])
        end_indexing = time.time()
        end_memory = get_memory_usage_mb()

        t0 = time.time()
        D, I = index.search(xq, k)
        t1 = time.time()
        print(f"데이터 {(i + 1) * 100000}개:")
        print(
            f"색인: {(end_indexing - start_indexing) * 1000 :.3f} ms ({end_memory - start_memory:.3f} MB) 검색: {(t1 - t0) * 1000 / nq :.3f} ms")

    print("\n## 예제 12.4 파라미터 m의 변경에 따른 성능 확인")
    k = 1
    d = xq.shape[1]
    nq = 1000
    xq = xq[:nq]

    for m in [8, 16, 32, 64]:
        index = faiss.IndexHNSWFlat(d, m)
        time.sleep(3)
        start_memory = get_memory_usage_mb()
        start_index = time.time()
        index.add(xb)
        end_memory = get_memory_usage_mb()
        end_index = time.time()
        print(f"M: {m} - 색인 시간: {end_index - start_index} s, 메모리 사용량: {end_memory - start_memory} MB")

        t0 = time.time()
        D, I = index.search(xq, k)
        t1 = time.time()

        recall_at_1 = np.equal(I, gt[:nq, :1]).sum() / float(nq)
        print(f"{(t1 - t0) * 1000.0 / nq:.3f} ms per query, R@1 {recall_at_1:.3f}")

    print("\n## 예제 12.5 ef_construction을 변화시킬 때 성능 확인")
    k = 1
    d = xq.shape[1]
    nq = 1000
    xq = xq[:nq]

    for ef_construction in [40, 80, 160, 320]:
        index = faiss.IndexHNSWFlat(d, 32)
        index.hnsw.efConstruction = ef_construction
        time.sleep(3)
        start_memory = get_memory_usage_mb()
        start_index = time.time()
        index.add(xb)
        end_memory = get_memory_usage_mb()
        end_index = time.time()
        print(
            f"efConstruction: {ef_construction} - 색인 시간: {end_index - start_index} s, 메모리 사용량: {end_memory - start_memory} MB")

        t0 = time.time()
        D, I = index.search(xq, k)
        t1 = time.time()

        recall_at_1 = np.equal(I, gt[:nq, :1]).sum() / float(nq)
        print(f"{(t1 - t0) * 1000.0 / nq:.3f} ms per query, R@1 {recall_at_1:.3f}")

    print("\n## 예제 12.6 ef_search 변경에 따른 성능 확인")
    for ef_search in [16, 32, 64, 128]:
        index.hnsw.efSearch = ef_search
        t0 = time.time()
        D, I = index.search(xq, k)
        t1 = time.time()

        recall_at_1 = np.equal(I, gt[:nq, :1]).sum() / float(nq)
        print(f"{(t1 - t0) * 1000.0 / nq:.3f} ms per query, R@1 {recall_at_1:.3f}")
