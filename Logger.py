class Logger():
    def __init__(self, file_name: str):
        directory = "./logs/" + file_name + ".txt"
        self.output_file = open(directory, "w+")
        print("Logger initialized...")

    def add_to_log(self, episode: int, reward: float, average_reward: float, loss: float):
        self.output_file.write(
            f"episode: {episode} | reward: {reward} | average reward: {average_reward} | loss: {loss}\n")
