import numpy

from hopfield.DataProcess import Dataset


def add_noise(x: numpy.ndarray, threshold: float = 0) -> numpy.ndarray:
	flip_mask: numpy.ndarray = numpy.random.rand(*x.shape) < threshold

	flipped_x = x.copy()
	flipped_x[flip_mask] *= -1

	return flipped_x


def signal_function(output: numpy.ndarray, state: numpy.ndarray) -> numpy.ndarray:
	if state.shape != output.shape:
		raise ValueError("signal_function: s and x should be in the same shape.")

	return numpy.where(output > 0, 1, numpy.where(output < 0, -1, state))


class HopfieldNetwork:
	def __init__(self, input_size: int, noise_rate: float = 0, seed: int = 42) -> None:
		numpy.random.seed(seed)

		self.__input_size: int = input_size
		self.__noise_rate: float = noise_rate
		self.__weight: numpy.ndarray = numpy.zeros((input_size, input_size))
		self.__threshold: numpy.ndarray = numpy.zeros((input_size))

	def memorize(self, knowledges: Dataset) -> None:
		for knowledge in knowledges:
			obscure_knowledge: numpy.ndarray = add_noise(knowledge, self.__noise_rate)
			self.__weight += numpy.outer(obscure_knowledge, obscure_knowledge)
		self.__weight /= self.__input_size
		numpy.fill_diagonal(self.__weight, 0)

		self.__threshold = self.__weight.sum(0)

	def recall(self, stimulate: numpy.ndarray, max_iter: int = 100) -> numpy.ndarray:
		state: numpy.ndarray = stimulate.copy()
		for iteration in range(max_iter):
			previous_state: numpy.ndarray = state.copy()

			for i in numpy.random.permutation(self.__input_size):
				output_i = numpy.dot(self.__weight[i, :], state) - self.__threshold[i]
				state[i] = signal_function(numpy.array([output_i]), numpy.array([state[i]]))[0]

			if numpy.array_equal(state, previous_state):
				print(f"Recall Iteration: {iteration + 1}")
				break

		return state
