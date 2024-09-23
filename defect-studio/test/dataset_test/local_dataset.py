from datasets import load_dataset, DatasetDict, concatenate_datasets

# 데이터셋 로드 (gzguevara/mr_potato_head_masked, gzguevara/cat_toy_masked)
mr_potato_head_masked = load_dataset("gzguevara/mr_potato_head_masked")
cat_toy_masked = load_dataset("gzguevara/cat_toy_masked")

# 첫 번째 마스크만 사용하도록 필터링하는 함수
def filter_first_mask(example):
    example["mask"] = example["mask_0"]  # mask_0만 사용
    return example

# 각 데이터셋에 대해 첫 번째 마스크만 필터링 적용
mr_potato_head_masked = mr_potato_head_masked.map(filter_first_mask)
cat_toy_masked = cat_toy_masked.map(filter_first_mask)

# 로컬 경로에 저장 (원하는 경로로 변경 가능)
mr_potato_head_masked.save_to_disk("./output/mr_potato_head_masked_local")
cat_toy_masked.save_to_disk("./output/cat_toy_masked_local")

print("데이터셋이 성공적으로 로컬에 저장되었습니다.")
