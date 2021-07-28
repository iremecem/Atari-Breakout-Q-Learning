from operator import ne
import numpy as np
import random
from collections import deque
from ImageModel import ImageModelHandler
from tensorflow.keras.optimizers import RMSprop


class Agent:

    __MEM_SIZE__: int = 100000 # Prev: 1000, 10000
    __GAMMA__: float = 0.99 # Prev: 0.1, 0.9, 0.95
    __MIN_E__: float = 0.1
    __E_INTERVAL__: float = 0.995 # Prev 0.9, 0.99

    def __init__(self, input_shape, num_actions):
        self.__INPUT_SHAPE__ = input_shape
        self.__NUM_ACTIONS__ = num_actions

        self.replayMemory = deque(maxlen=self.__MEM_SIZE__)
        self.epsilon = 1.0

        self.handler = ImageModelHandler()
        self.main_model = self.handler.createModel(
            self.__INPUT_SHAPE__, self.__NUM_ACTIONS__)
        self.target_model = self.handler.createModel(
            self.__INPUT_SHAPE__, self.__NUM_ACTIONS__)
        self.target_model.set_weights(self.main_model.get_weights())
        self.main_model.summary()

    def getReplayMemorySize(self):
        return len(self.replayMemory)

    def update(self):
        self.handler.updateModel(self.main_model, self.target_model)

    def save(self, name: str):
        self.handler.save(name, self.main_model)

    def load(self, name: str, typ:str):
        if typ == "train":
            self.main_model = self.handler.load(name)
            self.main_model.compile(loss='mse', optimizer=RMSprop(lr=0.00025, rho=0.95, epsilon=None, decay=0.0))
        elif typ == "test":
            self.main_model = self.handler.load(name)

    def addToReplayMemory(self, state, action, reward, next_state, done):
        self.replayMemory.append((state, action, reward, next_state, done))

    def predict(self, state):
        return self.main_model.predict(state)

    def step(self, state, is_train: bool):
        if is_train:
            if np.random.rand() <= self.epsilon:
                return random.randrange(self.__NUM_ACTIONS__)
            state = np.array([state])
            predictions = self.main_model.predict(state)
            return np.argmax(predictions[0])
        else:
            state = np.array([state])
            predictions = self.main_model.predict(state)
            predicted = np.argmax(predictions[0])
            print(f"Predicted: {predicted}")
            return predicted

    def replay(self, batch_size: int, strategy: str):
        batch = random.sample(self.replayMemory, batch_size)
        losses = []
        for state, action, reward, next_state, done in batch:
            state = np.array([state])
            next_state =np.array([next_state])
            if done:
                target = reward
            else:
                if strategy == "dqn":
                    target_predictions = self.target_model.predict(next_state)
                    target = (reward + self.__GAMMA__ *
                              np.amax(target_predictions))
                elif strategy == "ddqn":
                    main_predictions = self.main_model.predict(next_state)[0]
                    max_action = np.argmax(main_predictions)

                    target_predictions = self.target_model.predict(next_state)
                    target = (reward + self.__GAMMA__ *
                              target_predictions[0][max_action])
            labels = self.main_model.predict(state)
            labels[0][action] = target
            history = self.main_model.fit(state, labels, epochs=1, verbose=0)
            losses.extend(history.history["loss"])

        if self.epsilon > self.__MIN_E__:
            self.epsilon *= self.__E_INTERVAL__
        return np.average(losses) * 1e3 #Normalize
