import numpy

from hopfield.DataProcess import Dataset


def signal_function(output: numpy.ndarray, state: numpy.ndarray) -> numpy.ndarray:
	if state.shape != output.shape:
		raise ValueError("signal_function: s and x should be in the same shape.")

	return numpy.where(output > 0, 1, numpy.where(output < 0, -1, state))


class HopfieldNetwork:
	def __init__(self, input_size: int, seed: int = 42) -> None:
		numpy.random.seed(seed)

		self.__input_size: int = input_size
		self.__weight: numpy.ndarray = numpy.zeros((input_size, input_size))
		self.__threshold: numpy.ndarray = numpy.zeros((input_size))

	def memorize(self, knowledges: Dataset) -> None:
		for knowledge in knowledges:
			self.__weight += numpy.outer(knowledge, knowledge)
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
