from typing import List, Tuple

import numpy


def extract_dataset(path: str) -> List[numpy.ndarray]:
	dataset: List[numpy.ndarray] = []
	pattern: List[List[int]] = []
	with open(path, "r", encoding="utf-8") as file:
		lines: List[str] = file.readlines() + ["\n"]
		max_length: int = max([len(line.strip()) for line in lines])

		for line in lines:
			if line == "\n":
				dataset.append(numpy.array(pattern))
				pattern.clear()
				continue

			sub_pattern: str = line.strip()
			pattern.append([1 if c == "1" else 0 for c in sub_pattern] + [0] * (max_length - len(sub_pattern)))

	return dataset


class Dataset:
	def __init__(self, path: str) -> None:
		self.__dataset: List[numpy.ndarray] = extract_dataset(path)
		self.pattern_shape: Tuple[int, int] = self.__dataset[0].shape

	def __len__(self) -> int:
		return len(self.__dataset)

	def __getitem__(self, index) -> numpy.ndarray:
		return self.__dataset[index].flatten()
